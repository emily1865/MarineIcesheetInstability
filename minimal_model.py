import numpy as np
import matplotlib.pyplot as plt

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
	plt.savefig('bedrock.pdf')

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
		H_f = np.max([alpha_f*np.sqrt(L[i]), -eps*delta*d(L[i])]) # m
	
		# mass balance
		F = np.min([0,c*d(L[i])*H_f])
		B = a[i]*L[i]
	
		dLdt = 2*(B+F)/(3*alpha_m)*L[i]**-0.5 # m/a
	
		# new glacier length
		L[i+1] = L[i] + dLdt * (t[i+1]-t[i])
		
	if plot:
		# plot results	
		fig = plt.figure(figsize = (8,12))
		ax1 = fig.add_subplot(311)
		ax1.plot(t,L/1000, label = "L")
		ax1.xaxis.grid(True)
		ax1.set_xlim([0,5000])
		ax1.set_ylabel("L [km]")
		ax1.legend(loc = 2)
		ax2 = ax1.twinx()
		ax2.plot(t,a,linestyle = 'dashed', color = 'red', label ="a")
		ax2.set_ylabel("a [m ice a$^{-1}$]")
		ax2.legend(loc = 4)
		
		ax3 = fig.add_subplot(312)
		ax3.plot(t, np.max([alpha_f*np.sqrt(L), -eps*delta*d(L)],axis = 0), label = "H$_f$")
		ax3.set_ylabel("H$_f$, d$_f$ [m]")
		ax3.plot(t, [-np.min([0,d(Li)]) for Li in L], label = "d$_f$", linestyle = 'dashed', color = 'red')
		ax3.set_xlim([0,5000])
		ax3.xaxis.grid(True)
		ax3.legend()
		
		ax4 = fig.add_subplot(313)
		ax4.plot(t,a*L, label = "B")
		ax4.plot(t,[-np.min([0,c*d(Li)*np.max([alpha_f*np.sqrt(Li), -eps*delta*d(Li)])]) for Li in L], label = "F", linestyle = 'dashed', color = 'red')
		ax4.legend()
		ax4.set_xlabel("time [yrs]")
		ax4.set_ylabel("B, -F m$^2$ a$^{-1}$")
		ax4.set_xlim([0,5000])
		ax4.xaxis.grid(True)
		plt.savefig('case_1.pdf')
		
	return L
	
def case_2(L0, P_E, time, eps = 1, delta = 1.127, alpha_m = 2, alpha_f = 0.5, c = 2.4, beta = 0.005, plot = True):
	# L0: initial glacier length in m
	# E: equilibrium line height in m
	# time: time model should be run in years
	
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

	# time 
	timestep = 1 # year
	t = np.arange(0,time,timestep) # years
	
	# initialize variables
	L = np.zeros(t.shape)

	# inital glacier length
	L[0] = L0 # m
	
	# equilibrium line
	E = E0 + A_E*np.sin(2*np.pi*t/P_E + np.pi/2)
	
	# time loop
	for i in range(t.shape[0]-1):
		
		# front height
		H_f = np.max([alpha_f*np.sqrt(L[i]), -eps*delta*d(L[i])]) # m
	
		# mean height
		H_m = alpha_m*np.sqrt(L[i]) # m	
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
		fig = plt.figure(figsize = (8,12))
		ax1 = fig.add_subplot(311)
		ax1.plot(t,L/1000, label = "L")
		ax1.xaxis.grid(True)
		ax1.set_xlim([0,5000])
		ax1.set_ylabel("L [km]")
		ax1.legend(loc = 2)
		ax2 = ax1.twinx()
		ax2.plot(t,E,linestyle = 'dashed', color = 'red', label ="E")
		ax2.set_ylabel("E [m]")
		ax2.legend(loc = 4)
			
		ax3 = fig.add_subplot(312)
		ax3.plot(t, np.max([alpha_f*np.sqrt(L), -eps*delta*d(L)],axis = 0), label = "H$_f$")
		ax3.set_ylabel("H$_m$, H$_f$, d$_f$ [m]")
		ax3.plot(t, [-np.min([0,d(Li)]) for Li in L], label = "d$_f$", linestyle = 'dashed', color = 'red')
		ax3.plot(t, alpha_m*np.sqrt(L), linestyle = 'dotted', label = "H$_m$")
		ax3.set_xlim([0,5000])
		ax3.xaxis.grid(True)
		ax3.legend()
		
		B = beta*((d(0)+d(L)+np.max([alpha_f*np.sqrt(L), -eps*delta*d(L)], axis = 0)+alpha_m*np.sqrt(L))/2-E)*L
		F = np.array([np.min([0,c*d(Li)*np.max([alpha_f*np.sqrt(Li), -eps*delta*d(Li)])]) for Li in L])
		ax4 = fig.add_subplot(313)
		ax4.plot(t,B, label = "B")
		ax4.plot(t,F, label = "F", linestyle = 'dashed', color = 'red')
		ax4.plot(t, B+F, label = "B$_{tot}$", linestyle = 'dotted')
		ax4.legend()
		ax4.set_xlabel("time [yrs]")
		ax4.set_ylabel("B, F m$^2$ a$^{-1}$")
		ax4.set_xlim([0,5000])
		ax4.xaxis.grid(True)
		plt.savefig('case_2.pdf')
		
	return L
	
def dynamic_bedrock(tau, eps = 1, delta = 1.127, alpha_m = 2, alpha_f = 0.5, c = 2.4, beta = 0.005):
	# tau: time constant of bedrock in years
	
	# parameters of glacier
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
	t = np.arange(0,5000) # years, time step = 1 yr
	
	#horizontal coordinates in 10m steps
	x = np.arange(0,50000,10) #m
	
	# equilibrium line
	E = E0 + A_E*np.sin(2*np.pi*t/P_E + np.pi/2)
	
	# initialize variables
	L = np.zeros(t.shape)
	delta_d = np.zeros((t.shape[0],x.shape[0]))

	# inital glacier length
	L[0] = 0.0001 # m
	
	# inital mean slope
	s = -np.mean((d(x[1:])-d(x[0:-1]))/(x[1:]-x[0:-1]))
	
	# initial ice thickness
	H_f = np.max([alpha_f*np.sqrt(L[0]), -eps*delta*d(L[0])]) # m
	H_m = alpha_m*np.sqrt(L[0]) # m	
	C = 9/(4*L[0])*(H_m-H_f-d(L[0])-s*L[0]/2-np.mean(d(x)))**2
	H = np.nan_to_num(H_f + d(L[0]) + s*(L[0]-x) + np.sqrt(C*(L[0]-x))) # for x>L: H=0 --> replace nan from sqrt with 0
	
	# initial depression assuming isostatic equilibrium: initial bedrock height d+delta_d
	delta_d[0,:] = -rho*H
	
	# time loop
	for i in range(t.shape[0]-1):
		
		index_L = np.where(x==np.around(L[i],decimals = -1))[0] # use nearest neighbour approx
		
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
		
		C = 9/(4*L[i])*(H_m-d(L[i])-delta_d[i,index_L]-H_f-s*L[i]/2 - np.mean(d(x)+delta_d[i,:]))**2
		H = np.nan_to_num(H_f + d(L[i])+delta_d[i,index_L] + s*(L[i]-x) + np.sqrt(C*(L[i]-x))) 
		
		ddelta_ddt = -1/tau*(rho*H + delta_d[i])
		delta_d[i+1] = delta_d[i] + ddelta_ddt * (t[i+1]-t[i])
	
	return L, delta_d

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
	plt.savefig('case_1_hysteresis.pdf')
	
def case_2_hysteresis():
	# plot hysteresis curve for different periods of E
	
	E0 = 100 # m
	A_E = 350 # m
	P_E = 5000 # years
	E = E0 + A_E*np.sin(2*np.pi*np.arange(0,P_E)/P_E + np.pi/2)
	L2 = case_2(0.0001,P_E,P_E, plot=False)
	
	fig, ax = plt.subplots()
	plt.plot(E,L2/1000,label='$P_E = 5 kyr$')
	P_E = 10000 # years
	E = E0 + A_E*np.sin(2*np.pi*np.arange(0,P_E)/P_E + np.pi/2)
	L2 = case_2(0.0001,P_E,P_E, plot=False)
	plt.plot(E,L2/1000, linestyle = 'dashed',label='$P_E = 10 kyr$')
	P_E = 50000 # years
	E = E0 + A_E*np.sin(2*np.pi*np.arange(0,P_E)/P_E + np.pi/2)
	L2 = case_2(0.0001,P_E,P_E, plot=False)
	plt.plot(E,L2/1000, linestyle = 'dotted',label='$P_E = 50 kyr$')
	plt.xlabel(" E(m)")
	plt.ylabel("L [km]")
	plt.legend()
	ax.annotate(' ', xy = (430,25),xytext =(420,30),arrowprops=dict(arrowstyle="->"))
	ax.annotate(' ',xy =(-220,25),xytext=(-190,20),arrowprops=dict(arrowstyle="->"))
	plt.savefig('case_2_hysteresis.pdf')
	
def main():
	
	### PLOT BEDROCK ###
	#horizontal coordinates
	#x = np.linspace(0,50000, num = 100) #m
	#plot_bedrock(x)
	
	### CASE 1 ###
	# time 
	t = np.arange(0,5000) # years
	#accumulation
	a = 0.0005*t
	
	L1 = case_1(0.01,t,a)
	
	case_1_hysteresis()
	
	### CASE 2 ###
	L2 = case_2(0.0001,5000,5000)
	
	#hysteresis with different periods
	case_2_hysteresis()
	
	### DYNAMIC BEDROCK ###
	tau = 2000 # years, time constant for bedrock adjustment
	L_dyn, delta_d = dynamic_bedrock(tau)
	
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(L_dyn/1000, label = "L")
	ax1.xaxis.grid(True)
	ax1.set_xlim([0,5000])
	ax1.set_ylabel("L [km]")
	ax1.legend(loc = 3)
	ax2 = ax1.twinx()
	ax2.plot(np.mean(delta_d, axis = 1),linestyle = 'dashed', color = 'red', label ="mean depression")
	ax2.set_ylabel("<$\Delta$d> [m]")
	ax2.legend(loc = 1)	
	
if __name__ == "__main__":
    main()
