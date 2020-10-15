import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import pandas as pd

def d(x,d0 = 200, s =0.014, lam =300,xs = 40000,sigma = 10000):
	# bedrock elevation
	# d0,lam,xs,sigma in meters
	# returnsd in m
	return d0 -s*x+lam*np.exp(-((x-xs)/sigma)**2)

def plot_bedrock(x):
	#plot elevation
	plt.figure()
	plt.plot(x/1000,d(x))
	plt.xlabel("x [km]")
	plt.ylabel("d(x) [m]")
	plt.title("bedrock elevation")
	plt.savefig('img/bedrock.pdf')

def calculate_H_m(L, alpha_m):
	# mean ice thickness
	return alpha_m * np.sqrt(L)

def calculate_d_f(L):
	# water depth
	# not for dynamic bedrock
	return np.array([-np.min([0,d(Li)]) for Li in L])

def calculate_H_f(L, alpha_f, eps, delta):
	# frontal ice thickness
	# not for dynamic bedrock
	return np.max([alpha_f*np.sqrt(L), -eps*delta*d(L)], axis = 0)

def calculate_F(L, c, alpha_f, eps, delta):
	# calculate calving flux
	# not for dynamic bedrock
	return np.array([np.min([0,c*d(Li)*calculate_H_f(Li, alpha_f, eps, delta)]) for Li in L])
	
def calculate_h_m(L, alpha_f, eps, delta, alpha_m):
	# calculate mean glacier height
	# not for dynamic bedrock
	return (d(0)+d(L)+calculate_H_f(L, alpha_f, eps,delta) + calculate_H_m(L, alpha_m))/2

def case_1(L0,t,a, eps = 1, delta = 1.127, alpha_m = 2, alpha_f = 0.7, c = 2.4, plot = True):
	# L0: inital glacier length in m
	# t: time in years
	# a accumulation at times t in m/yr
	# parameters of glacier:
	# eps
	# delta
	# alpha_m in m**0.5
	# alpha_f in m**0.5
	# c = 2.4 in 1/a
	
	# initialize variables
	L = np.zeros(t.shape)

	# inital glacier length and accumulation rate
	L[0] = L0 # m
	
	# time loop
	for i in range(t.shape[0]-1):
		
		# front height
		H_f = calculate_H_f(L[i], alpha_f, eps, delta) # m
	
		# mass balance
		F = np.min([0,c*d(L[i])*H_f])
		B = a[i]*L[i]
	
		dLdt = 2*(B+F)/(3*alpha_m)*L[i]**-0.5 # m/a
	
		# new glacier length
		L[i+1] = L[i] + dLdt * (t[i+1]-t[i])
		
	if plot:
		# plot results	
		plot_case_1(t,L,a,eps,delta,alpha_m, alpha_f,c,'case_1.pdf')
	return L

def plot_case_1(t,L,a,eps,delta,alpha_m, alpha_f,c, name):
	# plot results for case 1
	
	fig = plt.figure(figsize = (8,10))
	ax1 = fig.add_subplot(311)
	ax1.plot(t,L/1000, label = "L")
	ax1.xaxis.grid(True)
	ax1.set_xlim([0,t[-1]])
	ax1.set_ylabel("L [km]")
	ax1.legend(loc = 2)
	ax2 = ax1.twinx()
	ax2.plot(t,a,linestyle = 'dashed', color = 'red', label ="a")
	ax2.set_ylabel("a [m ice a$^{-1}$]")
	ax2.legend(loc = 4)
	
	ax3 = fig.add_subplot(312)
	ax3.plot(t, calculate_H_f(L, alpha_f, eps, delta), label = "H$_f$")
	ax3.set_ylabel("H$_f$, d$_f$ [m]")
	ax3.plot(t, calculate_d_f(L), label = "d$_f$", linestyle = 'dashed', color = 'red')
	ax3.set_xlim([0,t[-1]])
	ax3.xaxis.grid(True)
	ax3.legend()
	
	ax4 = fig.add_subplot(313)
	ax4.plot(t,a*L, label = "B")
	ax4.plot(t,-calculate_F(L, c, alpha_f, eps, delta), label = "F", linestyle = 'dashed', color = 'red')
	ax4.legend()
	ax4.set_xlabel("time [yrs]")
	ax4.set_ylabel("B, -F m$^2$ a$^{-1}$")
	ax4.set_xlim([0,t[-1]])
	ax4.xaxis.grid(True)
	plt.savefig('img/case_1.pdf')
	
def case_1_hysteresis():
	#plot hysteresis curve
	
	t = np.arange(0,5000) # years
	a = 0.0005*t
	
	L1 = case_1(0.0001,t,a,plot=False)
	
	# hysteresis
	L1_hyst = case_1(10000, t,a, plot = False)
	L1_reverse = case_1(L1[-1],t,a[::-1], plot = False)
	
	fig,ax = plt.subplots()
	plt.plot(a, L1_hyst/1000)
	plt.plot(a,L1_reverse[::-1]/1000,c='C0')
	plt.xlabel("a [m ice a$^-1$]")
	plt.ylabel("L [km]")
	plt.ylim(0,50)
	ax.annotate(' ', xy = (1.3,19),xytext =(1.1,17),arrowprops=dict(arrowstyle="->"))
	ax.annotate(' ',xy =(1.1,46.5),xytext=(1.3,46.7),arrowprops=dict(arrowstyle="->"))
	plt.savefig('img/case_1_hysteresis.pdf')

def case_2(L0, P_E, t, eps = 1, delta = 1.127, alpha_m = 2, alpha_f = 0.5, c = 2.4, beta = 0.005, plot = True):
	# L0: initial glacier length in m
	# E: equilibrium line height in m
	# t: time array in years
	# parameters of glacier
	# eps
	# delta 
	# alpha_m in m**0.5
	# alpha_f in m**0.5
	# c in 1/a
	# beta in years
		
	# equilibirum line parameters
	E0 = 100 # m
	A_E = 350 # m
	#P_E = 5000 # years

	# initialize variables
	L = np.zeros(t.shape)

	# inital glacier length
	L[0] = L0 # m
	
	# equilibrium line
	E = E0 + A_E*np.sin(2*np.pi*t/P_E + np.pi/2)
	
	# time loop
	for i in range(t.shape[0]-1):
		
		# front height
		H_f = calculate_H_f(L[i], alpha_f, eps, delta) # m
	
		# mean height
		H_m = calculate_H_m(L[i], alpha_m)# m	
		h_m = (d(0)+d(L[i])+H_m+H_f)/2 # m
		
		# mass balance
		F = np.min([0,c*d(L[i])*H_f])
		B = beta*(h_m-E[i])*L[i]
	
		dLdt = 2*(B+F)/(3*alpha_m)*L[i]**-0.5 # m/a
	
		# new glacier length
		L[i+1] = L[i] + dLdt * (t[i+1]-t[i])
		
		# prevent glacier length from becoming negative
		if L[i+1]<=0:
			L[i+1] = 0.0001
	
	if plot:
		# plot results	
		fig = plt.figure(figsize = (8,10))
		ax1 = fig.add_subplot(311)
		ax1.plot(t,L/1000, label = "L")
		ax1.xaxis.grid(True)
		ax1.set_xlim([0,t[-1]])
		ax1.set_ylabel("L [km]")
		ax1.legend(loc = 2)
		ax2 = ax1.twinx()
		ax2.plot(t,E,linestyle = 'dashed', color = 'red', label ="E")
		ax2.set_ylabel("E [m]")
		ax2.legend(loc = 4)
			
		ax3 = fig.add_subplot(312)
		ax3.plot(t, calculate_H_f(L,alpha_f, eps, delta), label = "H$_f$")
		ax3.set_ylabel("H$_m$, H$_f$, d$_f$ [m]")
		ax3.plot(t, calculate_d_f(L), label = "d$_f$", linestyle = 'dashed', color = 'red')
		ax3.plot(t, calculate_H_m(L, alpha_m), linestyle = 'dotted', label = "H$_m$")
		ax3.set_xlim([0,t[-1]])
		ax3.xaxis.grid(True)
		ax3.legend()
		
		B = beta*(calculate_h_m(L, alpha_f, eps, delta, alpha_m)-E)*L
		F = calculate_F(L, c, alpha_f, eps, delta)
		ax4 = fig.add_subplot(313)
		ax4.plot(t,B, label = "B")
		ax4.plot(t,F, label = "F", linestyle = 'dashed', color = 'red')
		ax4.plot(t, B+F, label = "B$_{tot}$", linestyle = 'dotted')
		ax4.legend()
		ax4.set_xlabel("time [yrs]")
		ax4.set_ylabel("B, F m$^2$ a$^{-1}$")
		ax4.set_xlim([0,t[-1]])
		ax4.xaxis.grid(True)
		plt.savefig('img/case_2.pdf')
		
	return L

def case_2_hysteresis():
	# plot hysteresis curve for different periods of E
	
	E0 = 100 # m
	A_E = 350 # m
	P_E = 5000 # years
	t = np.arange(0,P_E)
	E = E0 + A_E*np.sin(2*np.pi*t/P_E + np.pi/2)
	L2 = case_2(0.0001,P_E,t, plot=False)
	
	fig, ax = plt.subplots()
	plt.plot(E,L2/1000,label='$P_E = 5 kyr$')
	P_E = 10000 # years
	t = np.arange(0,P_E)
	E = E0 + A_E*np.sin(2*np.pi*t/P_E + np.pi/2)
	L2 = case_2(0.0001,P_E,t, plot=False)
	plt.plot(E,L2/1000, linestyle = 'dashed',label='$P_E = 10 kyr$')
	P_E = 50000 # years
	t = np.arange(0,P_E)
	E = E0 + A_E*np.sin(2*np.pi*t/P_E + np.pi/2)
	L2 = case_2(0.0001,P_E,t, plot=False)
	plt.plot(E,L2/1000, linestyle = 'dotted',label='$P_E = 50 kyr$')
	plt.xlabel(" E(m)")
	plt.ylabel("L [km]")
	plt.legend()
	ax.annotate(' ', xy = (430,25),xytext =(420,30),arrowprops=dict(arrowstyle="->"))
	ax.annotate(' ',xy =(-220,25),xytext=(-190,20),arrowprops=dict(arrowstyle="->"))
	plt.savefig('img/case_2_hysteresis.pdf')
	
def dynamic_bedrock(tau, eps = 1, delta = 1.127, alpha_m = 2, alpha_f = 0.5, c = 2.4, beta = 0.005, plot = True):
	# tau: time constant of bedrock in years
	
	# parameters of glacier:
	# eps
	# delta 
	# alpha_m in m**0.5
	# alpha_f in m**0.5
	# c in 1/a
	# beta in years
	
	E0 = 100 # m
	A_E = 350 # m
	P_E = 5000 # years
	
	rho = 1/3 # ratio rho_ice/rho_rock
	
	# time 
	t = np.arange(0,5000,1) # years, time step = 1 yr
	
	#horizontal coordinates in 10m steps
	x = np.arange(0,50000,10) #m
	
	# equilibrium line
	E = E0 + A_E*np.sin(2*np.pi*t/P_E + np.pi/2)
	
	# initialize variables
	L = np.zeros(t.shape)
	delta_d = np.zeros((t.shape[0],x.shape[0]))
	h = np.zeros((t.shape[0],x.shape[0]))

	# inital glacier length
	L[0] = 0.0001 # m
	
	# inital mean slope
	s = -np.mean((d(x[1:])-d(x[0:-1]))/(x[1:]-x[0:-1]))
	
	# initial ice thickness
	H_f = np.max([alpha_f*np.sqrt(L[0]), -eps*delta*d(L[0])]) # m
	H_m = alpha_m*np.sqrt(L[0]) # m	
	C = 9/(4*L[0])*(H_m-H_f-d(L[0])-s*L[0]/2+np.mean(d(x[0])))**2
	H = np.nan_to_num(H_f + d(L[0]) + s*(L[0]-x) + np.sqrt(C*(L[0]-x))-d(x)) # for x>L: H=0 --> replace nan from sqrt with 0
	
	h[0,:] = H+d(x)
	# initial depression assuming isostatic equilibrium: initial bedrock height d+delta_d
	delta_d[0,:] = -rho*H
	
	# time loop
	for i in range(t.shape[0]-1):
		
		index_L = (L[i]/10).astype(int) #np.where(x==np.around(L[i],decimals = -1))[0] # use nearest neighbour approx
		
		# front height
		H_f = np.max([alpha_f*np.sqrt(L[i]), -eps*delta*(d(L[i])+delta_d[i,index_L])]) # m
	
		# mean height
		H_m = alpha_m*np.sqrt(L[i]) # m	
		h_m = (d(0)+delta_d[i,0] + d(L[i])+delta_d[i,index_L] + H_m+H_f)/2 # m
		
		# mass balance
		F = np.min([0,c*(d(L[i])+delta_d[i,index_L])*H_f])
		B = beta*(h_m-E[i])*L[i]
	
		dLdt = 2*(B+F)/(3*alpha_m)*L[i]**-0.5 # m/a
	
		# new glacier length
		L[i+1] = L[i] + dLdt * (t[i+1]-t[i])
		
		# prevent glacier length from becoming negative
		if L[i+1]<=0:
			L[i+1] = 0.0001
			
		#bedrock adjustment
		s = -np.mean((d(x[1:])+delta_d[i,1:]-d(x[0:-1])-delta_d[i,1:])/(x[1:]-x[0:-1]))
		
		C = 9/(4*L[i])*(H_m-d(L[i])-delta_d[i,index_L]-H_f-s*L[i]/2 + np.mean(d(x[0:index_L])+delta_d[i,0:index_L]))**2
		H = np.nan_to_num(H_f + d(L[i])+delta_d[i,index_L] + s*(L[i]-x) + np.sqrt(C*(L[i]-x)) - d(x)-delta_d[i,:])
		
		h[i,:] = H + d(x)+delta_d[i,:]
		
		ddelta_ddt = -1/tau*(rho*H + delta_d[i])
		delta_d[i+1] = delta_d[i] + ddelta_ddt * (t[i+1]-t[i])

	if plot:
		fig = plt.figure(figsize = (8,10))
		ax1 = fig.add_subplot(311)
		p1, = ax1.plot(t,L/1000, label = "L")
		ax1.xaxis.grid(True)
		ax1.set_xlim([0,t[-1]])
		ax1.set_ylabel("L [km]")
		ax2 = ax1.twinx()
		p2, = ax2.plot(t,np.mean(delta_d, axis = 1),linestyle = 'dashed', color = 'red', label ="mean depression")
		ax2.set_ylabel("<$\Delta$d> [m]")
		ax2b = ax1.twinx()
		ax2b.set_ylabel("E [m]")
		p2b, =ax2b.plot(t,E,label = 'E', color = 'darkorange',linestyle = 'dotted')
		ax2b.spines['right'].set_position(('outward', 60))
		ax1.legend(handles = [p1,p2,p2b])
		
		delta_d_f = delta_d_of_L(delta_d,L,x) 
		ax3 = fig.add_subplot(312)
		H_f = calculate_H_f_dynamic_bedrock(L,delta_d_f, alpha_f,eps,delta)
		ax3.plot(t, H_f, label = "H$_f$")
		ax3.set_ylabel("H$_m$, H$_f$, d$_f$ [m]")
		ax3.plot(t, calculate_d_f_dynamic_bedrock(L,delta_d_f), label = "d$_f$", linestyle = 'dashed', color = 'red')
		ax3.plot(t, calculate_H_m(L, alpha_m), linestyle = 'dotted', label = "H$_m$")
		ax3.set_xlim([0,t[-1]])
		ax3.xaxis.grid(True)
		ax3.legend()
		
		B = beta*((d(0)+delta_d[:,0]+d(L)+delta_d_f+H_f+alpha_m*np.sqrt(L))/2-E)*L
		F = np.array([np.min([0,c*(d(L[i])+delta_d_f[i])*H_f[i]]) for i in range(L.shape[0])])
		ax4 = fig.add_subplot(313)
		ax4.plot(t,B, label = "B")
		ax4.plot(t, F, label = "F", linestyle = 'dashed', color = 'red')
		ax4.plot(t, B+F, label = "B$_{tot}$", linestyle = 'dotted')
		ax4.legend()
		ax4.set_xlabel("time [yrs]")
		ax4.set_ylabel("B, F m$^2$ a$^{-1}$")
		ax4.set_xlim([0,t[-1]])
		ax4.xaxis.grid(True)
		plt.savefig('img/dynamic_bedrock.pdf')
	
	return L, x, delta_d, h 

def delta_d_of_L(delta_d,L,x):
	# get delta d at glacier front at L
	return np.array([delta_d[i,(L[i]/10).astype(int)] for i in range(L.shape[0])])

def calculate_H_f_dynamic_bedrock(L,delta_d_f, alpha_f,eps,delta):
	return np.max([alpha_f*np.sqrt(L), -eps*delta*(d(L)+delta_d_f)],axis = 0)

def calculate_d_f_dynamic_bedrock(L,delta_d_f):
	return np.array([-np.min([0,d(L[i])+delta_d_f[i]]) for i in range(L.shape[0])])

def calving(t, dO18,ocean_forcing, L0, file_name, tau=5000, eps = 1, delta = 1.127, alpha_m = 2, alpha_f = 0.5, c0 = 2.4, beta = 0.005, plot = True):
	# tau: time constant of bedrock in years
	# t: time array in years
	# dO18
	# ocean_forcing: tempterature change of ocean temperature
	# L0: inital glacier length
	
	# parameters of glacier:
	# eps
	# delta 
	# alpha_m in m**0.5
	# alpha_f in m**0.5
	# c0 in 1/a (proportionality parameter for calving)
	# beta in years
	
	rho = 1/3 # ratio rho_ice/rho_rock
	
	#horizontal coordinates in 10m steps
	x = np.arange(0,55000,10) #m
	
	#equilibrium line 
	E0 = 100
	dO180 = -41.429 # middle between max and min <-> mean: -41.59
	#dO18_today = -36.65
	#const = 78.5 # -> 350m amplitude
	E = E0 + 224.2*(dO18-dO180) # 224.2: amplitude 2000m = 13K T difference
	
	# calving constant
	c1 = c0*2.0
	c = c0 + c1*ocean_forcing
	
	# initialize variables
	L = np.zeros(t.shape)
	delta_d = np.zeros((t.shape[0],x.shape[0]))
	h = np.zeros((t.shape[0],x.shape[0]))

	# inital glacier length
	L[0] = L0 # m
	
	# inital mean slope
	s = -np.mean((d(x[1:])-d(x[0:-1]))/(x[1:]-x[0:-1]))
	
	# initial ice thickness
	H_f = np.max([alpha_f*np.sqrt(L[0]), -eps*delta*d(L[0])]) # m
	H_m = alpha_m*np.sqrt(L[0]) # m	
	C = 9/(4*L[0])*(H_m-H_f-d(L[0])-s*L[0]/2+np.mean(d(x[0])))**2
	H = np.nan_to_num(H_f + d(L[0]) + s*(L[0]-x) + np.sqrt(C*(L[0]-x))-d(x)) # for x>L: H=0 --> replace nan from sqrt with 0
	
	h[0,:] = H+d(x)
	# initial depression assuming isostatic equilibrium: initial bedrock height d+delta_d
	delta_d[0,:] = -rho*H
	
	# time loop
	for i in range(t.shape[0]-1):
		
		index_L = (L[i]/10).astype(int) #np.where(x==np.around(L[i],decimals = -1))[0] # use nearest neighbour approx
		
		# front height
		H_f = np.max([alpha_f*np.sqrt(L[i]), -eps*delta*(d(L[i])+delta_d[i,index_L])]) # m
	
		# mean height
		H_m = alpha_m*np.sqrt(L[i]) # m	
		h_m = (d(0)+delta_d[i,0] + d(L[i])+delta_d[i,index_L] + H_m+H_f)/2 # m
		
		# mass balance
		F = np.min([0,c[i]*(d(L[i])+delta_d[i,index_L])*H_f])
		B = beta*(h_m-E[i])*L[i]
	
		dLdt = 2*(B+F)/(3*alpha_m)*L[i]**-0.5 # m/a
	
		# new glacier length
		L[i+1] = L[i] + dLdt * (t[i+1]-t[i])
		
		# prevent glacier length from becoming negative
		if L[i+1]<=0:
			L[i+1] = 0.0001
			
		#bedrock adjustment
		s = -np.mean((d(x[1:])+delta_d[i,1:]-d(x[0:-1])-delta_d[i,1:])/(x[1:]-x[0:-1]))
		
		C = 9/(4*L[i])*(H_m-d(L[i])-delta_d[i,index_L]-H_f-s*L[i]/2 + np.mean(d(x[0:index_L])+delta_d[i,0:index_L]))**2
		H = np.nan_to_num(H_f + d(L[i])+delta_d[i,index_L] + s*(L[i]-x) + np.sqrt(C*(L[i]-x)) - d(x)-delta_d[i,:])
		
		h[i,:] = H + d(x)+delta_d[i,:]
		
		ddelta_ddt = -1/tau*(rho*H + delta_d[i])
		delta_d[i+1] = delta_d[i] + ddelta_ddt * (t[i+1]-t[i])
	
	if plot:
		fig = plt.figure(figsize = (8,10))
		ax1 = fig.add_subplot(312)
		p1, = ax1.plot(t,L/1000, label = "L")
		ax1.xaxis.grid(True)
		ax1.set_xlim([0,t[-1]])
		ax1.set_ylabel("L [km]")
		ax1.legend(loc = 2)
		ax2 = ax1.twinx()
		p2, = ax2.plot(t,np.mean(delta_d, axis = 1),linestyle = 'dashed', color = 'red', label ="mean depression")
		ax2.set_ylabel("<$\Delta$d> [m]")
		ax2.legend()
		
		ax3 = fig.add_subplot(311)
		ax3.plot(t,E, label = "E")
		ax3.set_ylabel("E [m]")
		ax3.legend(loc = 2)
		#ax3.set_ylim([-48,-36])
		ax3.set_xlim([0,t[-1]])
		ax3.xaxis.grid(True)
		ax3b = ax3.twinx()
		ax3b.plot(t,ocean_forcing, label = 'ocean forcing', color = 'red', linestyle = 'dashed')
		ax3b.set_ylabel('ocean forcing [°C]')
		ax3b.set_ylim([0,15])
		ax3b.legend()
	
		delta_d_f = delta_d_of_L(delta_d,L,x) 
		H_f = calculate_H_f_dynamic_bedrock(L,delta_d_f, alpha_f,eps,delta)
		B = beta*((d(0)+delta_d[:,0]+d(L)+delta_d_f+H_f+alpha_m*np.sqrt(L))/2-E)*L
		F = np.array([np.min([0,c[i]*(d(L[i])+delta_d_f[i])*H_f[i]]) for i in range(L.shape[0])])
		ax4 = fig.add_subplot(313)
		ax4.plot(t,F, label = "F")
		ax4.plot(t,F+B, label = "B$_{tot}$", linestyle = 'dashed', color = 'red', linewidth = 1)
		ax4.plot(t, B, label = "B", linestyle = 'dotted', linewidth = 1)
		ax4.legend(ncol = 3)
		ax4.set_xlabel("time [yrs]")
		ax4.set_ylabel("B, F m$^2$ a$^{-1}$")
		ax4.set_xlim([0,t[-1]])
		ax4.xaxis.grid(True)
		plt.savefig(file_name)
		
	return L, x, delta_d, h

def ocean_forcing(t, t_i, t_d=1000, T_max=2):
	# returns temperature change of ocean
	
	# t: time
	# t_i: time of beginning of event i
	# t_d: duration of events
	# T_max: maximum temperature change in °C
	
	delta_T = np.zeros(t.shape)
	for ti in t_i:
		peak = T_max*np.sin(2*np.pi*(t-ti)/(2*t_d))
		peak[t<ti]=0
		peak[t>(ti+t_d)]=0
		delta_T += peak
		
	return delta_T

def plot_paleo_record(age, dO18, ocean_forcing):
	
	fig = plt.figure(figsize = (12,5))
	ax1 = fig.add_subplot(111)
	ax1.plot(age[::-1], dO18[::-1], label = "$\delta^{18}$O")
	ax1.set_xlim([60000,10000])
	ax1.set_ylim([-48,-33])
	ax1.fill_betweenx([-48,33],46900,48300,color = 'lightgray', alpha = 0.5)
	ax1.fill_betweenx([-48,33],38190,39990,color = 'lightgray', alpha = 0.5)
	ax1.fill_betweenx([-48,33],28900,30600,color = 'lightgray', alpha = 0.5)
	ax1.fill_betweenx([-48,33],23340,25340,color = 'lightgray', alpha = 0.5)
	ax1.fill_betweenx([-48,33],14750,17200,color = 'lightgray', alpha = 0.5)
	ax1.set_ylabel("$\delta^{18}$O")
	ax1.set_xlabel("time [yrs b2k] (GICC05)")
	ax1.legend(loc = 2)
	ax2 = ax1.twinx()
	ax2.plot(age[::-1], ocean_forcing[::-1], color = 'red', label ="ocean forcing")
	ax2.set_ylabel("ocean forcing (°C)")
	ax2.set_ylim([0,15])
	ax2.legend(loc = 4)
	plt.savefig("img/paleo_record.pdf")

def dO18_to_temperature(dO18):
	# returns temperature in °C
	alpha_fit = 0.35
	beta_fit = 4
	T = (dO18+35.1)/alpha_fit + 241.6 + beta_fit
	return T -273.15
	
def make_movie(x,delta_d,h, filename):
	
	for i in range(0,h.shape[0],10):
		fig = plt.figure()
		plt.fill_between(x/1000, d(x)+delta_d[i,:], color = 'cornflowerblue')
		plt.fill_between(x/1000, 700, color = 'lightcyan')
		plt.fill_between(x/1000,d(x)+delta_d[i,:],h[i,:], color = 'white')
		plt.fill_between(x/1000,-400, d(x)+delta_d[i,:],color = 'lightgray')
		plt.xlabel('x [km]')
		plt.ylabel('height [m]')
		plt.xlim([0,x[-1]/1000])
		plt.ylim([-400,700])
		fig.savefig('movie/movie_%04d.png' %i)
		plt.close(fig)

	img_array = []
	for filename in glob.glob('movie/movie*.png'):
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)

	out = cv2.VideoWriter('movie/'+filename+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, size)
	
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()
	
	for file_name in glob.glob("movie/*.png"):
		os.remove(file_name)
	
def main():
	
	### PLOT BEDROCK ###
	#horizontal coordinates
	#x = np.linspace(0,50000, num = 100) #m
	#plot_bedrock(x)
	
	### CASE 1 ###
	# time 
	t = np.arange(0,5000) # years
	# accumulation
	a = 0.0005*t
	
	L1 = case_1(0.01,t,a)
	
	case_1_hysteresis()
	
	### CASE 2 ###
	L2 = case_2(0.0001,5000,t)
	
	# hysteresis with different periods
	case_2_hysteresis()
	
	### DYNAMIC BEDROCK ###
	tau = 2000 # years, time constant for bedrock adjustment
	L_dyn, x, delta_d, h = dynamic_bedrock(tau)
	#make_movie(x,delta_d,h,'dynamic_bedrock')
	
	### INFLUENCE OF OCEAN TEMPERATURE ON CALVING FOR CONSTANT EQUILIBRIUM HEIGHT ###
	t = np.arange(0,10000)
	delta_T_ocean = ocean_forcing(t,[2000,8000])
	L_calv, x, delta_d, h = calving(t,np.ones(t.shape)*(-43), delta_T_ocean, 40000, "img/ocean_forcing.pdf") # constant atmoph forcing: E0 = -200
	
	### PALEOCLIMATIC RECORDS ###
	# read data
	paleo_data = pd.read_excel("NGRIP_d18O_and_dust_5cm.xls",sheet_name = "NGRIP-2 d18O and Dust").to_numpy()
	age = paleo_data[:,3]
	dO18 = paleo_data[:,1]
	
	#smooth data, otherwise there is too much noise to see anything
	dO18_smooth = np.convolve(dO18, np.ones(15)/15, mode='valid')
	age_smooth = age[7:-7]
	
	#timing of ocean forcing, maximum right before aprupt warming
	t_i_paleo = np.array([14750,23340,27780,28900,32550,33830,35500,38190,40160, 41450,43460, 46900, 54250,55820])-500
	delta_T_ocean_paleo = ocean_forcing(age_smooth,t_i_paleo)
	
	# plot
	plot_paleo_record(age_smooth, dO18_smooth, delta_T_ocean_paleo)
	
	surface_T = dO18_to_temperature(dO18_smooth)
	
	# use only part of data and pay attention to time <-> age
	start = 9465 # = 25 kyrs
	end = 18424 # = 50 kyrs
	age_slice = age_smooth[start:end]
	dO18_slice = dO18_smooth[start:end]
	ocean_T_slice = delta_T_ocean_paleo[start:end]
	
	time = -(age_slice-age_slice[-1])[::-1]
	
	L_p, x_p, delta_d_p, h_p = calving(time,dO18_slice[::-1], ocean_T_slice[::-1], 5000, 'img/paleo_simulation.pdf')
	
	
if __name__ == "__main__":
    main()
