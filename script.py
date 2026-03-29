import pandas as pd


def fast_score_logic(df):
    # Очистка сумм (убираем пробелы и запятые)
    df['total_sum'] = df['total_sum'].replace(r'[\s\xa0]', '', regex=True).replace(',', '.', regex=True).astype(float)

    # Группировка по БИН — это фундамент Merit-based подхода
    scoring_db = df.groupby('bin').agg(
        total_amount=('total_sum', 'sum'),
        success_count=('status', lambda x: (x == 'Исполнена').sum()),
        total_apps=('status', 'count')
    ).reset_index()

    # Простая формула для MVP: надежность (50%) + масштаб (50%)
    scoring_db['reliability'] = scoring_db['success_count'] / scoring_db['total_apps']
    max_amount = scoring_db['total_amount'].max()
    scoring_db['scale'] = scoring_db['total_amount'] / max_amount

    scoring_db['final_score'] = (scoring_db['reliability'] * 50 + scoring_db['scale'] * 50).round(2)

    return scoring_db.sort_values('final_score', ascending=False)


print("Модуль первичного скоринга по БИН готов.")
