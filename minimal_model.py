import numpy as np
import matplotlib.pyplot as plt

def d(x,d0 = 200, s =0.014, lam =300,xs = 40000,sigma = 10000):
	# bedrock elevation
	# d0,lam,xs,sigma in meters
	return d0 -s*x+lam*np.exp(-((x-xs)/sigma)**2)

def main():

	#horizontal coordinates
	x = np.linspace(0,50000, num = 100) #m

	#plot elevation
	plt.figure()
	plt.plot(x/1000,d(x))
	plt.xlabel("x [km]")
	plt.ylabel("d(x) [m]")
	plt.title("bedrock elevation")
	plt.show()
	
	# parameters of glacier
	eps = 1
	delta = 1.127
	alpha_m = 2 # m**0.5
	alpha_f = 0.7 # m**0.5
	c = 2.4 # 1/a
	
	# time 
	t = np.linspace(0,5000,5000) # years
	
	# initialize variables
	L = np.zeros(t.shape)
	a = np.zeros(t.shape)

	# inital glacier length and accumulation rate
	L[0] = 1 # m
	a[0] = 0 # m/yr
	
	# time loop
	for i in range(t.shape[0]-1):
		
		# mean height
		h_m = alpha_m*np.sqrt(L[i]) # m
	
		# front height
		h_f = np.max([alpha_f*np.sqrt(L[i]), -eps*delta*d(L[i])]) # m
	
		# mass balance
		F = np.min([0,c*d(L[i])*h_f])
		B = a[i]*L[i]
	
		dLdt = 2*(B+F)/(3*alpha_m)*L[i]**-0.5 # m/a
	
		# new glacier length
		L[i+1] = L[i] + dLdt * (t[i+1]-t[i])
		
		# new accumulation rate
		a[i+1] = a[i]+0.0005
	
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
	plt.show()
	
if __name__ == "__main__":
    main()