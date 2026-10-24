"""
Model Versioning System for Adaptive GA Scalper
=============================================
Handles incremental model saving, archiving, loading, and rollback.

Greedy, opportunistic approach:
- Always save when improvement > threshold
- Archive previous versions
- Always use highest-fitness model
- Track full history for analysis
"""
import os, json, shutil, time, logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("versioning.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


MODEL_DIR = "models"
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
CURRENT_FILE = os.path.join(MODEL_DIR, "robust_params.json")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")
FITNESS_TRACKER = os.path.join(MODEL_DIR, "best_fitness_tracker.json")

MIN_IMPROVEMENT_THRESHOLD = 0.05
MAX_ARCHIVED_MODELS = 20
MAX_FITNESS_CAP = 250.0  # RobustGA realistic max ~100-150; old simple GA produced 1000+ fantasy values


@dataclass
class ModelVersion:
    version_id: str
    timestamp: str
    fitness: float
    profit: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    take_profit: float
    stop_loss: float
    max_hold: int
    cooldown: int
    momentum_short_period: int
    momentum_med_period: int
    momentum_short_threshold: float
    momentum_med_threshold: float
    validation_passed: bool
    regime_scores: Dict[str, float]


@dataclass
class FitnessTracker:
    best_fitness: float
    best_version_id: str
    total_improvements: int
    last_update: str
    history: List[Dict]


class ModelVersioningSystem:
    def __init__(self, model_dir: str = MODEL_DIR, 
                 min_improvement: float = MIN_IMPROVEMENT_THRESHOLD,
                 max_archived: int = MAX_ARCHIVED_MODELS):
        self.model_dir = model_dir
        self.archive_dir = ARCHIVE_DIR if model_dir == MODEL_DIR else os.path.join(model_dir, "archive")
        self.current_file = CURRENT_FILE if model_dir == MODEL_DIR else os.path.join(model_dir, "robust_params.json")
        self.metadata_file = METADATA_FILE if model_dir == MODEL_DIR else os.path.join(model_dir, "model_metadata.json")
        self.fitness_tracker_file = FITNESS_TRACKER if model_dir == MODEL_DIR else os.path.join(model_dir, "best_fitness_tracker.json")
        
        self.min_improvement = min_improvement
        self.max_archived = max_archived
        
        self._ensure_directories()
        self._init_metadata()
    
    def _ensure_directories(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
    
    def _init_metadata(self):
        if not os.path.exists(self.metadata_file):
            initial = {'versions': [], 'current_version': None, 'version_count': 0}
            with open(self.metadata_file, 'w') as f:
                json.dump(initial, f, indent=2)
            logger.info(f"Initialized metadata at {self.metadata_file}")
        
        if not os.path.exists(self.fitness_tracker_file):
            tracker = FitnessTracker(
                best_fitness=0.0,
                best_version_id=None,
                total_improvements=0,
                last_update=datetime.now().isoformat(),
                history=[]
            )
            self._save_fitness_tracker(tracker)
    
    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Metadata file is invalid JSON at {self.metadata_file}: "
                    f"line {e.lineno}, column {e.colno}: {e.msg}. Recreating empty metadata."
                )
                corrupt_backup = f"{self.metadata_file}.corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    shutil.copy2(self.metadata_file, corrupt_backup)
                    logger.info(f"Saved corrupt metadata backup to {corrupt_backup}")
                except Exception as copy_error:
                    logger.warning(f"Could not backup corrupt metadata file: {copy_error}")
                
                metadata = {'versions': [], 'current_version': None, 'version_count': 0}
                self._save_metadata(metadata)
                return metadata
        return {'versions': [], 'current_version': None, 'version_count': 0}
    
    def _save_metadata(self, metadata: Dict):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_fitness_tracker(self) -> FitnessTracker:
        if os.path.exists(self.fitness_tracker_file):
            with open(self.fitness_tracker_file) as f:
                d = json.load(f)
            return FitnessTracker(**d)
        return FitnessTracker(
            best_fitness=0.0,
            best_version_id=None,
            total_improvements=0,
            last_update=datetime.now().isoformat(),
            history=[]
        )
    
    def _save_fitness_tracker(self, tracker: FitnessTracker):
        with open(self.fitness_tracker_file, 'w') as f:
            json.dump(asdict(tracker), f, indent=2)
    
    def _generate_version_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = self._load_metadata()
        v_num = metadata['version_count'] + 1
        return f"v{v_num}_{ts}"
    
    def _create_archive_filename(self, version_id: str) -> str:
        return os.path.join(self.archive_dir, f"model_{version_id}.json")
    
    def save_model(self, params: Dict, metrics: Dict, validation_passed: bool = True) -> Tuple[bool, str]:
        new_fitness = params.get('fitness', 0.0)
        new_profit = metrics.get('total_profit', 0.0)
        new_pf = metrics.get('profit_factor', 0.0)
        new_wr = metrics.get('win_rate', 0.0)
        new_dd = metrics.get('max_drawdown', 0.0)
        regime_scores = metrics.get('regime_scores', {})
        if isinstance(regime_scores, dict):
            regime_scores = {k.name if hasattr(k, 'name') else str(k): v for k, v in regime_scores.items()}
        
        tracker = self._load_fitness_tracker()
        current_fitness = tracker.best_fitness
        
        # FIX: Calculate improvement_pct correctly for negative current fitness.
        # Old code gave `inf` when current_fitness <= 0, causing every negative-fitness
        # model to be saved as a "large improvement".
        if current_fitness > 0:
            improvement_pct = (new_fitness - current_fitness) / current_fitness
        else:
            improvement_pct = (new_fitness - current_fitness) / max(abs(current_fitness), 1.0)
        
        should_save = False
        reason = ""
        
        # CRITICAL FIX: Never save models with fitness <= 0 that also failed validation.
        # These are proven losing strategies in backtest AND out-of-sample.
        if not validation_passed and new_fitness <= 0:
            logger.info(f"Rejecting save: fitness={new_fitness:.1f} <= 0 and validation_failed")
            return False, None
        
        if current_fitness == 0:
            if new_fitness > 0:
                should_save = True
                reason = "First model (positive fitness)"
            else:
                # Don't seed the tracker with a losing first model
                logger.info(f"Rejecting first model: fitness={new_fitness:.1f} is not positive")
                return False, None
        elif validation_passed and improvement_pct >= self.min_improvement:
            should_save = True
            reason = f"Validation passed + Improvement: {improvement_pct:.1%}"
        elif validation_passed and new_profit > 0:
            should_save = True
            reason = f"Validation passed (PF={new_pf:.2f}, WR={new_wr:.1%})"
        elif not validation_passed and new_fitness > 0 and improvement_pct >= 0.30:
            # Only save failed-validation models if positive fitness and dramatically better
            should_save = True
            reason = f"Validation FAILED but large improvement: {improvement_pct:.1%}"
        
        # Overfit protection: never save models with unrealistic fitness
        if new_fitness > MAX_FITNESS_CAP:
            logger.warning(f"Rejecting overfit model save (fitness={new_fitness:.1f} > {MAX_FITNESS_CAP})")
            return False, None
        
        if not should_save:
            logger.info(f"No save: Fitness={new_fitness:.1f} (current best: {current_fitness:.1f}), "
                      f"improvement: {improvement_pct:.1%}")
            return False, None
        
        version_id = self._generate_version_id()
        metadata = self._load_metadata()
        
        if os.path.exists(self.current_file):
            old_metadata = metadata['current_version']
            if old_metadata:
                old_file = self._create_archive_filename(old_metadata['version_id'])
                try:
                    with open(self.current_file) as sf:
                        shutil.copyfileobj(sf, open(old_file, 'w'))
                    logger.info(f"Archived previous model: {old_metadata['version_id']}")
                except Exception as e:
                    logger.warning(f"Archive failed: {e}")
        
        model_version = ModelVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            fitness=new_fitness,
            profit=new_profit,
            profit_factor=new_pf,
            win_rate=new_wr,
            max_drawdown=new_dd,
            take_profit=params.get('take_profit', 15.0),
            stop_loss=params.get('stop_loss', 10.0),
            max_hold=params.get('max_hold', 600),
            cooldown=params.get('cooldown', 180),
            momentum_short_period=params.get('momentum_short_period', 4),
            momentum_med_period=params.get('momentum_med_period', 9),
            momentum_short_threshold=params.get('momentum_short_threshold', 0.06),
            momentum_med_threshold=params.get('momentum_med_threshold', 0.10),
            validation_passed=validation_passed,
            regime_scores=regime_scores
        )
        
        with open(self.current_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        metadata['versions'].append(asdict(model_version))
        metadata['current_version'] = asdict(model_version)
        metadata['version_count'] += 1
        self._save_metadata(metadata)
        
        tracker.best_fitness = new_fitness
        tracker.best_version_id = version_id
        tracker.total_improvements += 1
        tracker.last_update = datetime.now().isoformat()
        tracker.history.append({
            'version_id': version_id,
            'fitness': new_fitness,
            'profit': new_profit,
            'improvement': improvement_pct,
            'timestamp': datetime.now().isoformat()
        })
        self._save_fitness_tracker(tracker)
        
        self._prune_old_models()
        
        logger.info(f"=== MODEL SAVED: {version_id} ===")
        logger.info(f"  Fitness: {new_fitness:.1f} (prev: {current_fitness:.1f}, +{improvement_pct:.1%})")
        logger.info(f"  Profit: ${new_profit:.2f}, PF: {new_pf:.2f}, WR: {new_wr:.1%}")
        logger.info(f"  Reason: {reason}")
        
        return True, version_id
    
    def load_best(self) -> Optional[Dict]:
        if os.path.exists(self.current_file):
            with open(self.current_file) as f:
                params = json.load(f)
            
            # Overfit protection: reject models with unrealistic fitness
            model_fitness = params.get('fitness', 0)
            if model_fitness > MAX_FITNESS_CAP:
                logger.warning(f"Rejecting overfit model (fitness={model_fitness:.1f} > {MAX_FITNESS_CAP}). "
                             f"Clearing versioning data for fresh start.")
                self._reset_overfit_data(model_fitness)
                return None
            
            metadata = self._load_metadata()
            tracker = self._load_fitness_tracker()
            
            # FIX: Sync tracker with actual params file if they are out of sync.
            # This handles the case where robust_params.json was updated but the
            # fitness tracker still shows stale 0.0 from an old model.
            if tracker.best_fitness == 0.0 and model_fitness > 0:
                logger.info(f"Resyncing fitness tracker: tracker={tracker.best_fitness:.1f} -> params={model_fitness:.1f}")
                tracker.best_fitness = model_fitness
                tracker.best_version_id = metadata.get('current_version', {}).get('version_id') if metadata.get('current_version') else None
                self._save_fitness_tracker(tracker)
            elif abs(tracker.best_fitness - model_fitness) > 0.01 and model_fitness > tracker.best_fitness:
                logger.info(f"Resyncing fitness tracker: tracker={tracker.best_fitness:.1f} -> params={model_fitness:.1f}")
                tracker.best_fitness = model_fitness
                tracker.best_version_id = metadata.get('current_version', {}).get('version_id') if metadata.get('current_version') else None
                self._save_fitness_tracker(tracker)
            
            if metadata.get('current_version'):
                logger.info(f"Loaded best model: {metadata['current_version']['version_id']} "
                         f"(fitness={metadata['current_version']['fitness']:.1f})")
            else:
                logger.info(f"Loaded best model from params file (fitness={model_fitness:.1f})")
            return params
        return None
    
    def _reset_overfit_data(self, overfit_fitness: float):
        """Reset versioning data when overfit models are detected"""
        # Archive the overfit params file
        if os.path.exists(self.current_file):
            backup = self.current_file + f".overfit_{int(overfit_fitness)}"
            try:
                shutil.move(self.current_file, backup)
                logger.info(f"Archived overfit params to {backup}")
            except Exception as e:
                os.remove(self.current_file)
                logger.warning(f"Removed overfit params: {e}")
        
        # Reset fitness tracker
        tracker = FitnessTracker(
            best_fitness=0.0,
            best_version_id=None,
            total_improvements=0,
            last_update=datetime.now().isoformat(),
            history=[]
        )
        self._save_fitness_tracker(tracker)
        logger.info("Reset fitness tracker due to overfit detection")
    
    def load_specific_version(self, version_id: str) -> Optional[Dict]:
        archive_file = self._create_archive_filename(version_id)
        if os.path.exists(archive_file):
            with open(archive_file) as f:
                return json.load(f)
        
        metadata = self._load_metadata()
        for v in metadata.get('versions', []):
            if v['version_id'] == version_id:
                return {k: v[k] for k in ['take_profit', 'stop_loss', 'max_hold', 'cooldown',
                                          'momentum_short_period', 'momentum_med_period',
                                          'momentum_short_threshold', 'momentum_med_threshold',
                                          'fitness']}
        return None
    
    def rollback(self, version_id: str) -> bool:
        params = self.load_specific_version(version_id)
        if params is None:
            logger.error(f"Cannot rollback: version {version_id} not found")
            return False
        
        with open(self.current_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        metadata = self._load_metadata()
        for v in metadata['versions']:
            if v['version_id'] == version_id:
                metadata['current_version'] = asdict(ModelVersion(**v))
                break
        self._save_metadata(metadata)
        
        logger.info(f"Rolled back to: {version_id}")
        return True
    
    def get_model_history(self, limit: int = 10) -> List[Dict]:
        metadata = self._load_metadata()
        versions = metadata.get('versions', [])
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        return versions[:limit]
    
    def get_best_versions(self, n: int = 5) -> List[Dict]:
        metadata = self._load_metadata()
        versions = metadata.get('versions', [])
        versions.sort(key=lambda x: x['fitness'], reverse=True)
        return versions[:n]
    
    def _prune_old_models(self):
        metadata = self._load_metadata()
        versions = metadata.get('versions', [])
        
        if len(versions) <= self.max_archived:
            return
        
        versions.sort(key=lambda x: x['timestamp'])
        
        to_remove = versions[:-self.max_archived]
        for v in to_remove:
            archive_file = self._create_archive_filename(v['version_id'])
            if os.path.exists(archive_file):
                os.remove(archive_file)
                logger.info(f"Pruned old model: {v['version_id']}")
        
        metadata['versions'] = versions[-self.max_archived:]
        self._save_metadata(metadata)
    
    def get_stats(self) -> Dict:
        tracker = self._load_fitness_tracker()
        metadata = self._load_metadata()
        
        return {
            'total_versions': metadata['version_count'],
            'current_best': tracker.best_version_id,
            'best_fitness': tracker.best_fitness,
            'total_improvements': tracker.total_improvements,
            'last_update': tracker.last_update,
            'archived_count': len(os.listdir(self.archive_dir)) if os.path.exists(self.archive_dir) else 0
        }
    
    def print_summary(self):
        stats = self.get_stats()
        history = self.get_model_history(5)
        
        print("\n" + "=" * 60)
        print("MODEL VERSIONING SYSTEM - SUMMARY")
        print("=" * 60)
        print(f"Total versions: {stats['total_versions']}")
        print(f"Current best: {stats['current_best']}")
        print(f"Best fitness: {stats['best_fitness']:.1f}")
        print(f"Improvements: {stats['total_improvements']}")
        print(f"Last update: {stats['last_update']}")
        print(f"Archived: {stats['archived_count']}")
        print("\nRecent versions:")
        for v in history:
            print(f"  {v['version_id']}: Fitness={v['fitness']:.1f}, "
                  f"Profit=${v['profit']:.2f}, PF={v['profit_factor']:.2f}")
        print("=" * 60 + "\n")


def save_model_from_ga(params, metrics, validation_passed=True):
    vs = ModelVersioningSystem()
    return vs.save_model(params, metrics, validation_passed)


def load_best_model():
    vs = ModelVersioningSystem()
    return vs.load_best()


if __name__ == "__main__":
    vs = ModelVersioningSystem()
    
    print("Testing Model Versioning System...")
    print(f"Model dir: {vs.model_dir}")
    print(f"Archive dir: {vs.archive_dir}")
    
    if os.path.exists(vs.current_file):
        print("\nExisting models found:")
        vs.print_summary()
    else:
        print("\nNo models yet. System ready for first save.")