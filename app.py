from flask import Flask
app = Flask('app')
import numpy as np
from pprint import pprint
np.set_printoptions(precision=3)
np.set_printoptions(formatter={'int':lambda x : f'{x: 3}'})
 
import urllib.request as httpcl
import json
from datetime import date
from joblib import Parallel, delayed, cpu_count

def refresh_backdata():
    global lasttime
    global starttime
    global pre_resultstr
    global crawlNo
    
    if lasttime == get_recent_game_no():
        return
    print("#### refresh back data ####")
    lasttime = get_recent_game_no()
    starttime = lasttime-9
    pre_resultstr = ""
    crawlNo=[]
    for x in range(starttime, lasttime) :
        tempcrawl = get_lotto(x)
        print(tempcrawl)
        crawlNo.append(tempcrawl)
    sorted(crawlNo, key=lambda x : x['no'])
    pre_resultstr = pre_resultstr+ "\n".join([f"{x}" for x in crawlNo])+"\n"
    print(pre_resultstr)
    
    return

def print_lotto_beautiful(games):
    resultstr =""
    result_game_no  = len(games)
    printstr = [ "o"*45 for _ in range(result_game_no)]
    for idx, x in enumerate(games):
        for y in x:
            printstr[idx] = printstr[idx][:y-1]+'X'+printstr[idx][y:]
    column = 6
    for row_no in range((45//column)+1):
        row = [f"{x[row_no*column : row_no*column+column]:6}" for x in printstr]
        resultstr = resultstr + "  |  ".join(row) + "\n"
    return resultstr

def softmax(x):
    if np.sum(x < 0 ) >0:
        y = x - np.min(x)
    else :
        y = x
    return  y / np.sum(y)

def get_lotto(no):
    r= json.load(httpcl.urlopen(f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={no}"))
    if r['returnValue'] != 'success':
        return None,None
    crawl =  [r[f'drwtNo{x+1}'] for x in range(6)]
    crawl.sort()
    rstr = f"{r['drwNo']} {r['drwNoDate']} {crawl}"
    return_obj = {'no' : int(r['drwNo']), 'date_str':r['drwNoDate'], 'nums': crawl}
    return return_obj

def get_recent_game_no():
    startdate = date(2002,12,7)
    enddate = date.today()
    dday = (enddate - startdate).days
    lasttime = dday//7 +2
    return lasttime

def get_luck(choice_probs):
    return np.sort( np.random.choice(45 , size=6, replace=False, p=choice_probs) +1)

def get_lucky_nums(label, temp_choic_probs):
    resultstr = "#####"*8+"\n"
    
    resultstr = resultstr + f"{label}  ( min : {np.min(temp_choic_probs):.3f} , max : {np.max(temp_choic_probs):.3f} , sum : { np.sum(temp_choic_probs):.3f})\n"
    resultstr = resultstr + f"{temp_choic_probs}\n"

    result_game_no = 20000
    result = np.zeros((result_game_no,6), dtype=np.int64)

    for x in range(result_game_no):
        result[x] = get_luck(temp_choic_probs)
    temp =  np.sum((np.ones(1, dtype=np.int64)*2)**result, axis=1)
    result_nums, r_idx, case_cnt = np.unique(temp, return_counts=True, return_index= True)

    sorted_idx = np.argsort(case_cnt)[-5:]
    resultstr = resultstr + f"duplicate count : {case_cnt[sorted_idx]} / {np.sum(case_cnt)} \n"
    result = result[ r_idx[sorted_idx]]
    
    resultstr = resultstr + f"{result}\n"
    resultstr = resultstr + print_lotto_beautiful(result)
    resultstr = resultstr +"\n" +"#####"*8
    return resultstr

def get_probs():
    facts = [ x for x in range (46)]
    print(facts)
    for i in range(2,46):
        facts[i] = facts[i] * facts[i-1]
    print(facts)

    C44_6 = int(facts[44] / (facts[38]*facts[6]) )
    C44_5 = int(facts[44] / (facts[39]*facts[5]) )
    C45_6 = int(facts[45] / (facts[39]*facts[6]) )

    print(C44_6, C44_5,C44_6+ C44_5)
    print(C45_6)
    return (float(C44_5) / float(C45_6))

    
def get_lucky_number():
    global lasttime
    global starttime
    global pre_resultstr
    global crawlNo
    global prob_table
    resultstr = ""
    
    resultstr = resultstr+ f"{nums}" +"\n"
    resultstr = resultstr+ f"{case_cnt}" +"\n"

    history_cnt = np.array([x['nums'] for x in crawlNo[1-limit_game_gap:]])
    probs_bias = 0.2
    choice_probs = np.ones(45) * probs_bias
    history_num, history_cnt = np.unique(history_cnt, return_counts=True)
    for idx,x in enumerate(history_num):
        choice_probs[x-1] += prob_table[history_cnt[idx]-1]
    softmax_choice_probs = softmax(np.exp(choice_probs))
    softmax_non_exp_choice_probs = softmax(choice_probs)
    
    resultstr = resultstr+ "history_num, history_cnt" +"\n"
    resultstr = resultstr+ f"최근 {limit_game_gap-1}게임 동안 나온 숫자 종류 : {len(history_num)}, 변경이력 총 합 : {np.sum(history_cnt)}" +"\n"
    #resultstr = resultstr+ f"{' '.join([x for x in *zip(history_num,history_cnt )])}" +"\n"
    resultstr = resultstr+ f"choice_probs  ( bias : {probs_bias} , min : {np.min(choice_probs):.3f} , max : {np.max(choice_probs):.3f})" +"\n"
    resultstr = resultstr+ f"{choice_probs}" +"\n"

    #get_lucky_nums("softmax_choice_probs", softmax_choice_probs)
    resultstr = resultstr+ get_lucky_nums("softmax_non_exp_choice_probs", softmax_non_exp_choice_probs) +"\n"

    print(resultstr)
    return pre_resultstr+ resultstr


global lasttime
global starttime
global pre_resultstr
global crawlNo
global prob_table

lasttime = 1
starttime = lasttime
pre_resultstr = ""
crawlNo=[]
prob_table = np.zeros(9)
print("get probtable")
probs = get_probs() #1086008. / 8145060.  # 숫자 1이 포함될 확률
sample_cnt = 100000       # sample game 수
game_contain_no_1 = np.random.rand(sample_cnt)<= probs  # sample game 수 만큼 수행 했을때 1이 포함된 게임을 True 외엔 False
#print((aa<=probs)*1)
limit_game_gap = 9  # 최근 n 게임에 대한 통계..(카운트)
cumsum_game_cnt = np.cumsum(game_contain_no_1)
cumsum_game_cnt[limit_game_gap:] -= cumsum_game_cnt[:-limit_game_gap]
nums, case_cnt = np.unique(cumsum_game_cnt[game_contain_no_1], return_counts=True)
for idx,x in enumerate(nums):
    prob_table[x-1] = case_cnt[idx]
prob_table = prob_table/np.sum(prob_table)
print("get probtable end !!")
refresh_backdata()


@app.route('/')
def hello_world():
    refresh_backdata()
    resultstr = get_lucky_number()
    return f"<html><body><pre>{resultstr}</pre></body></html>"


if __name__ == "__main__" :
    get_lucky_number()
    #app.run(host='0.0.0.0', port=8080)




