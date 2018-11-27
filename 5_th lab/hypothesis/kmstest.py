import log
from math import exp, sqrt
from scipy.special import gamma, iv

def calcStat(seq, n, F):
	# Подсчет статистики критерия Крамера-Мизеса-Смирнова
	S = 0.0
	for i in range(n):
		S += (F(seq[i]) - (2 * i - 1) / (2 * n)) ** 2
	return 1 / (12 * n) + S

def calcSignif(stat):
	# Подсчет достигнутого уровня значимости
	a1 = 0.0; j = 0; part1 = part2 = part3 = 1.0
	#for j in range(5):
	while j < 5 and part1 != 0.0 and part2 != 0.0 and part3 != 0.0:
		part1 = (gamma(j + 1 / 2) * sqrt(4 * j + 1)) / (gamma(1 / 2) * gamma(j + 1))
		part2 = exp(-(4 * j + 1) ** 2 / (16 * stat))
		part3 = iv(-1 / 4, (4 * j + 1) ** 2 / (16 * stat)) - iv(1 / 4, (4 * j + 1) ** 2 / (16 * stat))
		a1 += part1 * part2 * part3
		j += 1
	a1 *= 1 / sqrt(2 * stat)
	return 1 - a1

def check(seq, n, F, alpha):
	# Критерий Крамера-Мизеса-Смирнова
	seq.sort()
	stat = calcStat(seq, n, F)
	signif = calcSignif(stat)
	log.write("stat = ", stat, "\n")
	log.write("signif = ", signif, "\n")
	return signif > alpha