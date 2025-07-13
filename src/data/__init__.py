# -*- coding: utf-8 -*-
"""
Data processing modules
"""

from .analysis import (
    load_lottery_data, check_data_quality, calculate_statistics,
    create_enhanced_features, plot_frequency_distribution,
    plot_hot_cold_numbers, plot_gap_statistics, plot_patterns,
    plot_trend_analysis
)
from .processing import (
    process_analysis_data, get_trend_features, prepare_recent_trend_data,
    format_quality_report, format_frequency_stats, format_hot_cold_stats,
    format_gap_stats, format_pattern_stats, format_trend_stats
)
from .statistics import (
    calculate_advanced_statistics, calculate_column_statistics,
    plot_advanced_statistics, plot_distribution_analysis
)

__all__ = [
    # Analysis module
    'load_lottery_data', 'check_data_quality', 'calculate_statistics',
    'create_enhanced_features', 'plot_frequency_distribution',
    'plot_hot_cold_numbers', 'plot_gap_statistics', 'plot_patterns',
    'plot_trend_analysis',
    # Processing module
    'process_analysis_data', 'get_trend_features', 'prepare_recent_trend_data',
    'format_quality_report', 'format_frequency_stats', 'format_hot_cold_stats',
    'format_gap_stats', 'format_pattern_stats', 'format_trend_stats',
    # Statistics module
    'calculate_advanced_statistics', 'calculate_column_statistics',
    'plot_advanced_statistics', 'plot_distribution_analysis'
]