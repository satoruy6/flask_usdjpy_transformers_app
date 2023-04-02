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
        import numpy as np
        import csv
        import math
        import pandas as pd
        import yfinance as yf
        from sklearn.pipeline import Pipeline 
        from sklearn.preprocessing import StandardScaler 
        from sklearn.linear_model import LogisticRegression

        # 外為データ取得
        tks  = 'USDJPY=X'
        data = yf.download(tickers  = tks ,          # 通貨ペア
                        period   = '1y',          # データ取得期間 15m,1d,1mo,3mo,1y,10y,20y,30y  1996年10月30日からデータがある。
                        interval = '1h',         # データ表示間隔 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                        )

        #最後の日時を取り出す。
        lastdatetime = data.index[-1]

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
        # print (successive_data)
        # print (answers)

        # データ数
        n = len(successive_data)
        # print (n)
        m = len(answers)
        # print (m)

        # pipeline(transformers)モデル
        clf = Pipeline([ ('scaler', StandardScaler()), ('clf', LogisticRegression())])
        # サポートベクターマシーンによる訓練 （データの75%を訓練に使用）
        clf.fit(successive_data[:int(n*750/1000)], answers[:int(n*750/1000)])

        # テスト用データ
        # 正解
        expected = answers[int(-n*250/1000):]
        # 予測
        predicted = clf.predict(successive_data[int(-n*250/1000):])

        predict_datetime=f'{lastdatetime}の次の1時間足の予測'
        # 末尾の10個を比較
        #print ('正解:' + str(expected[-10:]))
        #print ('予測:' + str(list(predicted[-10:])))

        # 正解率の計算
        correct = 0.0
        wrong = 0.0
        for i in range(int(n*250/1000)):
            if expected[i] == predicted[i]:
                correct += 1
            else:
                wrong += 1
                
        #print('正解数： ' + str(int(correct)))
        #print('不正解数： ' + str(int(wrong)))

        successive_data.append([modified_data[count_m-4], modified_data[count_m-3], modified_data[count_m-2], modified_data[count_m-1]])
        predicted = clf.predict(successive_data[-1:])
        #print ('次の1時間足の予測:' + str(list(predicted)) + ' 1:陽線,　0:陰線')
        if str(list(predicted)) == str([1]):
            predicted_result='「陽線」でしょう。'
        else:
            predicted_result='「陰線」でしょう。'
        accuracy="正解率: " + str(round(correct / (correct+wrong) * 100,  2)) + "%"   
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