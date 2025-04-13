#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Entry Point for Lottery Prophet Application
Author: Yang Zhao
"""

import os
import sys
import logging
import argparse
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),  
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():

    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    
    # 确保必要的目录存在
    for lottery_type in ['dlt', 'ssq']:
        model_dir = os.path.join('model', lottery_type)
        os.makedirs(model_dir, exist_ok=True)
        
        data_dir = os.path.join('data', lottery_type)
        os.makedirs(data_dir, exist_ok=True)

def fetch_data(lottery_type):
    """获取彩票数据"""
    logger.info(f"开始获取{lottery_type}彩票数据...")
    
    if lottery_type == 'dlt':
        from scripts.fetch_dlt_data import main as fetch_dlt
        fetch_dlt()
    elif lottery_type == 'ssq':
        from scripts.fetch_ssq_data import main as fetch_ssq
        fetch_ssq()
    else:
        logger.error(f"不支持的彩票类型: {lottery_type}")
        return False
    
    logger.info(f"{lottery_type}彩票数据获取完成")
    return True

def train_model(lottery_type, model_type):
    """训练彩票预测模型"""
    logger.info(f"开始训练{lottery_type}彩票的{model_type}模型...")
    
   
    from lottery_predictor_app_new import train_model as app_train_model
    
   
    def log_to_console(message):
        logger.info(message)
    

    success = app_train_model(lottery_type, model_type, log_callback=log_to_console)
    
    if success:
        logger.info(f"{lottery_type}彩票的{model_type}模型训练完成")
    else:
        logger.error(f"{lottery_type}彩票的{model_type}模型训练失败")
    
    return success

def predict(lottery_type, model_type):
    """使用训练好的模型进行预测"""
    logger.info(f"使用{model_type}模型预测{lottery_type}彩票...")
    
    from lottery_predictor_app_new import predict_next_draw as app_predict
    results = app_predict(lottery_type, model_type)
    
    if results:
        logger.info(f"预测结果: {results}")
    else:
        logger.error(f"预测失败")
    
    return results

def run_app():
    """运行完整的GUI应用程序"""
    logger.info("启动彩票预测应用程序...")
    
    from lottery_predictor_app_new import main as app_main
    app_main()

def main():
    """主函数，解析命令行参数并运行对应功能"""
    parser = argparse.ArgumentParser(description='彩票预测系统')
    

    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
   
    fetch_parser = subparsers.add_parser('fetch', help='获取彩票数据')
    fetch_parser.add_argument('lottery_type', choices=['dlt', 'ssq'], help='彩票类型')
    
 
    train_parser = subparsers.add_parser('train', help='训练预测模型')
    train_parser.add_argument('lottery_type', choices=['dlt', 'ssq'], help='彩票类型')
    train_parser.add_argument('--model', default='lightgbm', 
                             choices=['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble'],
                             help='模型类型')
    
  
    predict_parser = subparsers.add_parser('predict', help='预测下一期彩票号码')
    predict_parser.add_argument('lottery_type', choices=['dlt', 'ssq'], help='彩票类型')
    predict_parser.add_argument('--model', default='lightgbm',
                               choices=['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble'],
                               help='模型类型')
    
  
    app_parser = subparsers.add_parser('app', help='运行GUI应用程序')
    

    args = parser.parse_args()
    
    
    setup_environment()
    
 
    if args.command == 'fetch':
        return fetch_data(args.lottery_type)
    elif args.command == 'train':
        return train_model(args.lottery_type, args.model)
    elif args.command == 'predict':
        return predict(args.lottery_type, args.model)
    elif args.command == 'app':
        return run_app()
    else:
      
        logger.info("未指定子命令，默认启动GUI应用程序")
        return run_app()

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"程序开始运行: {start_time}")
        result = main()
        end_time = datetime.now()
        logger.info(f"程序结束运行: {end_time}")
        logger.info(f"总运行时间: {end_time - start_time}")
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"程序运行出错: {e}")
        sys.exit(1) 