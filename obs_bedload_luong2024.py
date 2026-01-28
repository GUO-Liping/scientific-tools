from seismic_bedload import SaltationModel, MultimodeModel

f = np.linspace(0.001, 20, 100)
D = 0.3  
sigma = 0.52
mu = 0.15
s = sigma/np.sqrt(1/3-2/np.pi**2)
pD = log_raised_cosine_pdf(D, mu, s)/D
D50 = 0.4
H = 4.0     
W = 50
theta = np.tan(1.4*np.pi/180)
r0 = 600
qb = 1e-3
model = SaltationModel()

# Forward modeling of PSD
psd = model.forward_psd(f, D, H, W, theta, r0, qb, D50 = D50, pdf = pD)

# Inverting  bedload flux
PSD_obs = np.loadtxt("PSD.txt")
H = np.loadtxt("flowdepth.txt")
bedload_flux = model.inverse_bedload(PSD_obs, f, D, H, W, theta, r0, qb, D50=D50, tau_c50=tau_c50, pdf = pD)