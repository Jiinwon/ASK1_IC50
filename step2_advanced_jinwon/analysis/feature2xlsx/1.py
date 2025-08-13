import pandas as pd

# 1) Parquet 읽기
path = "/home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/data/featurized/feature_7.parquet"
df = pd.read_parquet(path)

# 2) Excel로 저장
output = "/home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/data/featurized/feature_7.xlsx"
df.to_excel(output, index=False)

print(f"엑셀 파일로 저장 완료: {output}")
