# 历史号码检查功能更新

## 更新日期

2023年XX月XX日

## 更新内容

本次更新主要增加了历史号码检查功能，可以防止预测出与历史开奖号码过于相似的号码组合。具体功能包括：

1. **历史号码检查**：检查预测号码是否与历史号码重复或相似
2. **自动调整预测**：当预测号码与历史号码过于相似时，自动调整预测结果
3. **可配置的相似度规则**：支持自定义相似度规则，例如前区匹配5个、后区匹配0个等

## 功能说明

### 相似度规则

系统默认的相似度规则如下：

#### 双色球(SSQ)

- 前区6个相同，后区1个相同（完全相同）
- 前区6个相同，后区0个相同
- 前区5个相同，后区1个相同
- 前区5个相同，后区0个相同

#### 大乐透(DLT)

- 前区5个相同，后区2个相同（完全相同）
- 前区5个相同，后区1个相同
- 前区5个相同，后区0个相同
- 前区4个相同，后区2个相同

### 使用方法

在预测过程中，系统会自动检查预测号码是否与历史号码相似。如果相似，系统会自动调整预测结果，确保预测号码不会与历史号码过于相似。

如果您想要禁用历史号码检查功能，可以在调用预测函数时设置 `check_history=False`：

```python
# 禁用历史号码检查
prediction = process_predictions(red_predictions, blue_predictions, lottery_type, check_history=False)
```

如果您想要自定义相似度规则，可以在调用预测函数时设置 `similarity_rules` 参数：

```python
# 自定义相似度规则
similarity_rules = [
    {'front_match': 6, 'back_match': 1},  # 前区6个相同，后区1个相同
    {'front_match': 5, 'back_match': 0}    # 前区5个相同，后区0个相同
]
prediction = process_predictions(red_predictions, blue_predictions, lottery_type, similarity_rules=similarity_rules)
```

## 技术实现

本功能主要通过以下几个函数实现：

1. `check_prediction_against_history`：检查预测号码是否与历史号码相似
2. `filter_predictions_by_history`：过滤掉与历史号码相似的预测结果
3. `adjust_prediction_to_avoid_history`：调整预测号码，避免与历史号码相似

这些函数已经集成到预测处理流程中，您可以直接使用 `process_predictions` 和 `randomize_numbers` 函数，系统会自动检查和调整预测结果。

## 注意事项

1. 历史号码检查功能可能会增加预测的时间，但能有效避免预测出与历史相似的号码
2. 如果您需要更高的预测效率，可以禁用历史号码检查功能
3. 自定义相似度规则时，请确保规则合理，避免过于严格的规则导致无法生成有效的预测结果