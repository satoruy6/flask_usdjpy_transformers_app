from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # "予測開始"ボタンをクリックしたら処理する
    if request.method=='POST':
        import time
        t1 = time.time()
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        import numpy as np
        import csv
        import math
        import pandas as pd
        from pandas import Series, DataFrame
        import yfinance as yf

        # 外為データ取得
        tks  = 'USDJPY=X'
        data = yf.download(tickers  = tks ,          # 通貨ペア
                        period   = '1y',          # データ取得期間 15m,1d,1mo,3mo,1y,10y,20y,30y  1996年10月30日からデータがある。
                        interval = '1h',         # データ表示間隔 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                        )

        #最後の日時を取り出す。
        lastdatetime = data.index[-1]
        #print(lastdatetime)
        #Close価格のみを取り出す。
        data_close = data['Close']

        #対数表示に変換する
        ln_fx_price = []
        for line in data_close:
            ln_fx_price.append(math.log(line))
        count_s = len(ln_fx_price)

        # 為替の上昇率を算出、おおよそ-1.0-1.0の範囲に収まるように調整
        modified_data = []
        for i in range(1, count_s):
            modified_data.append(float(ln_fx_price[i] - ln_fx_price[i-1])*1000)
        count_m = len(modified_data)

        # 前日までの4連続の上昇率のデータ
        successive_data = []
        # 正解値 価格上昇: 1 価格下落: 0
        answers = []
        for i in range(4, count_m):
            successive_data.append([modified_data[i-4], modified_data[i-3], modified_data[i-2], modified_data[i-1]])
            if modified_data[i] > 0:
                answers.append(1)
            else:
                answers.append(0)
        #print(successive_data)
        #print(type(successive_data))
        #print(len(successive_data), "行", len(successive_data[0]), "列")
        # print (answers)

        df_successive_data = pd.DataFrame(successive_data)
        df_successive_data = df_successive_data.astype(float)
        df_successive_data.columns=['x__0','x__1','x__2','x__3']
        df_answers = pd.DataFrame(answers)
        df_answers.columns=['y']
        df_answers = df_answers['y'].astype(int)

        df = pd.concat([df_successive_data, df_answers], axis=1)
        n = len(df)
        #print(df)
        #dataset = df
        #dataset.to_csv('dataset.csv')


        # データの読み込み
        #df = pd.read_csv('dataset.csv')

        # 目的変数を定義
        y = df['y']

        # 説明変数を定義
        X = df.drop('y', axis=1)

        # 学習データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # パイプラインを作成
        pipe = make_pipeline(StandardScaler(), LogisticRegression())

        # パイプラインを用いてモデルを学習
        pipe.fit(X_train, y_train)

        #訓練済みのパイプライン（モデル）を保存する。
        #joblib.dump(pipe, 'pipe.pkl')

        # テストデータでモデルを評価
        score = pipe.score(X_test, y_test)
        score_round = round(score*100, 2)

        #最新のデータを加える
        successive_data.append([modified_data[count_m-4], modified_data[count_m-3], modified_data[count_m-2], modified_data[count_m-1]])
        df_successive_data = pd.DataFrame(successive_data)
        df_successive_data.columns=['x__0','x__1','x__2','x__3']
        data_latest = df_successive_data

        #予測をする
        y_pred = pipe.predict(data_latest)
        #print(y_pred[-10:])

        # 結果を表示
        accuracy=f'正解率：{score_round}%'
        predict_datetime=f'{lastdatetime}の次の1時間足の予測'
        if y_pred[-1:] >= 0.5:
            predicted_result='陽線でしょう'
        else:
            predicted_result='陰線でしょう'  
        t2 = time.time()
        elapsed_time = t2- t1
        elapsed_time = round(elapsed_time, 2)
        delay_time='プログラム処理時間： ' + str(elapsed_time) + '秒'

        return render_template('result.html', predict_datetime=predict_datetime, predicted_result=predicted_result, accuracy=accuracy, delay_time=delay_time)

@app.route('/reset', methods=['POST'])
def reset():
    # "予測開始"ボタンをクリックしたら処理する
    if request.method=='POST':
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)