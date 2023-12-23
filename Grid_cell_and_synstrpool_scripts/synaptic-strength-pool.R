synaptic_strength_pool = 
  {
    # setting seed here enforces uniformity across all possible simulations
    set.seed(432143)
  
    
    # empirical probability density function
    P = function(s) {
      A = 100.7
      B = .02
      sigma1 = .022
      sigma2 = .018
      sigma3 = .150
      A*(1-exp(-s/sigma1))*(exp(-s/sigma2)+B*exp(-s/sigma3))
    }
    
    # synaptic size to strength conversion
    W = function(s) (s/.2)*(s/(s+.0314))
    
    # generate a pool of synaptic strengths by rejection sampling
    s = runif(1000000, 0, .2)
    p = runif(1000000, 0, 23)
    synaptic_size_pool = s[p<=P(s)]
    synaptic_strength_pool = W(synaptic_size_pool)
    synaptic_strength_pool
  }

