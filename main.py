'''
A simple implementation of MH sampling

reference github repo: https://github.com/choo8/mcmc-algorithms

@Zeming 
'''
import os 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

# define the saving path
path = os.path.dirname(os.path.abspath(__file__))

# While estimating the posterior, it is usually expensive
# to calculate the intergral ∫_θ p(x,θ)dθ = p(x)

'''
EXAMPLE: 

Given the following generatic model:

        M,S --> μ --> x

The Gaussian distribution x ~ N( μ, σ=5) decided by its mean parameter
μ decided by a prior distribution μ ~ N( M, S).
'''

class example:

    def __init__( self):
        self.M = 60
        self.S = 10
        self.mu = np.random.normal( loc=self.M, scale=self.S, size=1)[0]
        self.sigma = 5
    
    def reset( self):
        self.mu = np.random.normal( loc=self.M, scale=self.S, size=1)[0]

def observation(env):

    # number of samples to draw
    num_samples = 2 ** np.arange(9)

    # show histogram of samples
    history = []
    _, axes = plt.subplots( 3,3, figsize=(10,10))
    plt.suptitle( f'Distribution of x given μ={env.mu:.4f}')
    for i, n_samples in enumerate(num_samples):
        
        # get samples 
        x_samples = np.random.normal( loc=env.mu, scale=env.sigma, size=n_samples)
        history.append( x_samples)

        # histogram to count the number of each samples
        axes[ i//3, i%3].hist(x_samples)
        axes[ i//3, i%3].set( xlabel='x', ylabel='#x', title=f'{n_samples} Data points')

    # save the figure
    try:
        plt.savefig( f'{path}/figures/observation.png')
    except:
        os.mkdir( 'figures')
        plt.savefig( f'{path}/figures/observation.png')

    return history

def analytical_solution( env, history):

    # inference the posterior based on the prior information
    # using Bayes rule: p(μ|x) ∝ p(x|μ)p(μ) where p(x|μ) = ∏_xi p(xi|μ)
    # p(μ|x) ~ N( (σ^2M + n S^2 x_mean)/(nS^2+σ^2), σ^2S^2/(nS^2+σ^2) 

    _, axes = plt.subplots( 3,3, figsize=(10,10))
    plt.suptitle( f'Infer the distribution p(μ|x), \n ground turth: {env.mu}')
    for i, x_samples in enumerate(history):
        
        # μ space
        n = len( x_samples)
        mus = np.linspace( 0, 100, 100)
        x_mean = np.mean(x_samples)
        posterior_mu_mean = ( env.sigma**2 * mus + n * env.S**2 * x_mean) \
                            / ( n * env.S**2 + env.sigma**2)
        posterior_mu_std  = ( env.sigma**2 * env.S**2 ) \
                            / ( n * env.S**2 + env.sigma**2)
        posterior_mu_pdf  = norm.pdf( mus, loc=posterior_mu_mean, scale=posterior_mu_std)

        # histogram to count the number of each samples
        axes[ i//3, i%3].plot( mus, posterior_mu_pdf)
        axes[ i//3, i%3].set( xlabel='x', ylabel='p(μ|x)', title=f'{n} Data points')

    # save the figure
    plt.savefig( f'{path}/figures/analytical_solution.png')

def MH_solution( env, history, burnin=1000, T=1000, std=3):
    '''Infer the posterior using MH sampling method

    I add a burnin period 
    '''

    # visualization
    _, axes = plt.subplots( 3,3, figsize=(10,10))
    plt.suptitle( f'''
                    Infer the distribution p(μ|x) using sampling, 
                    ground turth: {env.mu}
                   ''')

    for i, x_samples in enumerate(history):

        # initialize the first sample
        n = len(x_samples)
        curr_mu = env.M
        samples = [curr_mu]
        
        for t in range(1, burnin + T + 1):

            # propose a sample μ'~ q(μ'|μt)
            prop_mu = np.random.normal(curr_mu, std)

            # calculate the acceptance rate:
            # p(μ')q(μt|μ') = ∏_xi p(xi|μ')p(μ')q(μt|μ')
            upper = norm( prop_mu, env.sigma).pdf(x_samples).prod()\
                  * norm( env.M, env.S).pdf( prop_mu) \
                  * norm( curr_mu, std).pdf( prop_mu)
                
            # p(μt)q(μ'|μt) = ∏_xi p(xi|μt)p(μt)q(μ'|μt)
            lower = norm( curr_mu, env.sigma).pdf(x_samples).prod()\
                  * norm( env.M, env.S).pdf( curr_mu) \
                  * norm( prop_mu, std).pdf( curr_mu)
                
            # A = min(1, p(μ')q(μt|μ')/p(μt)q(μ'|μt))
            accept_rate = np.min( [ 1, upper/lower])

            # sample to decide acceptance or not 
            if np.random.rand() < accept_rate:
                curr_mu = prop_mu

            # if after burnin period, add to the sample pools 
            if t > burnin+1:
                samples.append( curr_mu)

        # use historgram to count the frequency and normalize to distribution
        bins = np.arange(40, 90, 2)
        freq, _ = np.histogram( samples, bins)
        freq = freq / np.sum( freq)

         # histogram to count the number of each samples
        xt = np.linspace( bins[0], bins[-2], 5)
        axes[ i//3, i%3].plot( bins[:-1], freq)
        axes[ i//3, i%3].set( xticks=xt, xlabel='x', ylabel='p(μ|x)', title=f'{n} Data points')

         # save the figure
        plt.savefig( f'{path}/figures/MH_solution.png')


if __name__ == '__main__':

    # prepare example
    env = example()
    history = observation(env)

    # analytical solution
    analytical_solution( env, history)

    # MH sampling 
    MH_solution( env, history)
