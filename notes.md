- Anomly Detection
 - identifying patterns in high freq data
 - detecting anomalous order flow helps idetnify market manipulation    
 - also helps idetnify data errors
 - useful for opportunity detection
    - IMBALANCE DETECTION
    - VOLUME SPIKE DETECTION
ML/DL Approaches
 - Supervised Learning
    - training models on labeled data (anoamlies are pre identified)
 - Unsupervised Learning
    - clustering techniques to group similar patterns
    - outliers in clusters are anomalies

GAN (Generative Adversarial Networks)
 - can generate normal market data
 - anomalies detected by comparing real data to generated data
 - Generator
 - Discriminator
 - to create new data 
 - The concept is to create an adversarial process where two neural networks compete against each other to improve their performance.

 1). Generator
 - The forger/artist 
    - creates synthetic data that mimics real market data
    - Input: random vector of noise
    - Output: synthetic market data (e.g., order book snapshots, trade sequences)
 2). Discriminator
    - The detective
    - evaluates both real market data and synthetic data from the generator
    - Input: market data (real or synthetic)
    - Output: probability score indicating whether the data is real or synthetic

Why GAN for stock modelling
- compatures complex distribiution
- vol clusters
- fat tails
- closesly mimics real data
- GANS are good in an unsupervised setting

Example pipeline
- Limit order book, exectuion data, for a specific symvol
- Normalize to stabalize data
- window sequencing (teemporal dependencies)
   - since the model learns time patterns, 
   - the continous data is segmented into fixed windows
- Training data selection

Step 2:
- Model arch
- use a recurrent GAN (LSTM or GRU layers to capture temporal dependencies)
- The G is the random noise vector Z,
The D is the fed both the real and the fake data
the two networks compete untile the generator is capable of producing sequences that the discrimnator can only calssify real with 50% probablity
- the outcome;
- the trained generator becomes an expert at reporducing normal data the set of all possible typical market
patterns

Step 3:
- Anomaly Scoring
- when a new unseen sequence of arrives the goal is to map the normal maniforal and measure how far it goes
1). Map to Latent Space
- the core idea is to find the best latent noise vector to find a fake sample that closely matches the real sample
2). Calculate score
-A(xtest) = L(R) = abs( xtest - G(zopt)) = Reconstroction Loss
- Discriminator Loss: the difference between the features extracted by the discriminator for the real and generated data: LD = abs( f(xtest) - f(G(zopt)) )
- Final Anomaly Score: A(xtest) = (1-λ) * L


4 Decision
 - Set a threshold based on validation data to see if scores are above 95th or 99th percentile
  - if A(xtest) > threshold => anomaly
  - Actions: flag for review, trigger alerts, further analysis


Score Definations
- 1). Discriminator functions (updates) - Detecting Fakes:
    - the goal is to be a perfect classifier
    - high prob for real data, low prob for fake data from the generator
    - L(D) measures both real and fake samples. This loss is minimized using backpropagation
    - if D correctly classifies real and Fake then LD decreases else increases
- 2) . Generators Updates
    - Maximize the probability that the disciminator will make a mistake
    - if D classifies the generators output as Real (G fools D) Lg decreases
    - if D classifies G oupput as fake Lg increases

Goes towards nash equilibrium
- G wants to generate data that perfectly mimics distribution
- D wants to perfectly distinguish the 2
- when equilibrium is reached, then traning stops
- At equilibrium, G has learned Orderbook distribution 
- D can no longer tell difference between fake and real data


