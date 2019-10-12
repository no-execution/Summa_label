from rouge import Rouge
import json
import re
import jieba
import time
import os
from multiprocessing import Pool,Process,Queue
import gc
import threading

def word_token(text):
	list_words = list(jieba.cut(text))
	return list_words


def test_process(inp_file,out_file):
	'''
	把一篇文章的topk个句子和尾句拿出来作为摘要

	Args:
		inp_file:输入
		out_file:输出
		k:取k个
	'''
	with open(inp_file,'r') as f:
		content = f.readlines()

	with open(out_file,'w') as writer:
		for line in content:
			#pair = line.split('\t')[1]
			sents = [' '.join(word_token(x.strip())) for x in re.split('[。！？～；...\u200b\xa0]',line.strip()) if len(x)>5]
			#target task需要的文本不能太短 对句子长度进行限制
			writer.write('。'.join(sents)+'\n')
			
			'''
			extract_sents = extract_sents.replace('网络配图','')
			extract_sents = extract_sents.replace('资料图','')
			extract_sents = extract_sents.replace('图片来源：','')
			extract_sents = extract_sents.replace('微博图片','')
			extract_sents = extract_sents.replace('图/新京报网','')
			extract_sents = extract_sents.replace('视频截图','')
			extract_sents = extract_sents.replace('网络图片','')
			'''

			#writer.write(extract_sents.strip()+'\n')
			#writer.write(pair.strip()+'\n')

def get_label(inp_file_text,inp_file_label,json_file):
	'''
	通过启发式方法对句子进行打标

	Args:
		inp_file_text:正文的文件路径
		inp_file_label:摘要的文件路径
		json_file:保存的json文件的路径

	输入文本无需事先分词
	'''
	f1 = open(inp_file_text,'r')
	f2 = open(inp_file_label,'r')
	r = Rouge()
	with open(json_file,'w',encoding='utf-8') as writer:
		count = 0
		start = time.time()
		for doc,summarize in zip(f1,f2):
			if (count+1)%1000 == 0:
				print('finished',count+1,'instances')
				print('time usage:',time.time()-start)
				start = time.time()
			sents = [' '.join(word_token(sent.strip())) for sent in re.split('[。！？～；...\u200b\xa0]',doc.strip()) if len(sent.strip())>=5]
			sents = [x for x in sents if x != '']
			labels = [0] * len(sents)
			summarize = ' '.join(word_token(summarize))
			#获取所有句子的rouge-1分数，并取出最大的那个
			rouge_group = []
			score_mid,max_idx = 0,0
			for i,sent_i in enumerate(sents):
				try:
					score = r.get_scores(sent_i,summarize)[0]['rouge-1']['f']
				except:
					print('sent_i:',sent_i)
					print('sents:',sents)
					print('summarize:',summarize)
					continue
					#return sents
				if score > score_mid:
					score_mid = score
					max_idx = i
			rouge_group.append(sents[max_idx])
			labels[max_idx] = 1
			#遍历剩余句子，如果加入group中，能使整个group的rouge score增加 
			#则label[id] = 1 并且把该句子加入group
			#否则continue
			#一定要注意句子的顺序以及label和sentence之间的对应关系
			score_max = score_mid
			for j,sent_j in enumerate(sents):
				if j == max_idx:
					continue
				score = r.get_scores(' '.join(rouge_group+[sent_j]),summarize)[0]['rouge-1']['f']
				if score > score_max:
					labels[j] = 1
					rouge_group.append(sent_j)
					score_max = score
			#保存结果
			dict_mid = {}
			dict_mid['doc'] = '\n'.join(sents)
			dict_mid['labels'] = '\n'.join([str(x) for x in labels])
			dict_mid['summaries'] = summarize

			writer.write(json.dumps(dict_mid,ensure_ascii=False)+'\n')
			count += 1

#多进程程序，如果处理的文件太大可以把文件分成多个小文件并行处理
def main():
	path = '/data/irong/summa/data/'
	start = time.time()
	p = Pool(10)       #i7 8750h 6核12线程
	for i in range(16):
		gc.disable()
		text,summa = path+'sent'+str(i)+'.txt',path+'summ'+str(i)+'.txt'
		p.apply_async(get_label,args=(text,summa,'out_'+str(i)+'.json',))
		print('Process ',i,'is running.......')
		gc.enable()
	p.close()
	p.join()
	end = time.time()
	print('time used :',end-start)

