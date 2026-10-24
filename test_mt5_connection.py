import MetaTrader5 as mt5
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    logger.info("Testing MT5 connection...")
    
    # Initialize MT5 connection
    if not mt5.initialize():
        logger.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    
    logger.info(f"MT5 Version: {mt5.version()}")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        mt5.shutdown()
        return False
    
    logger.info(f"Connected to MT5 account #{account_info.login}")
    logger.info(f"Balance: ${account_info.balance:.2f}")
    logger.info(f"Equity: ${account_info.equity:.2f}")
    
    # Shutdown connection
    mt5.shutdown()
    logger.info("MT5 connection test successful")
    return True

if __name__ == "__main__":
    test_connection()