import numpy as np
import random
from RecurRank import RecurRank
from TopRank import TopRank
from CascadeLinUCB import CascadeLinUCB
from CascadeLinTS import CascadeLinTS
from Environment import CasEnv, PbmEnv#, RealDataEnv
import time

def main(L,d,T,envname,repeat=1,synthetic=True,tabular=False,filename=''):

    for i in range(repeat):
        seed = int(time.time() * 100) % 399
        print("Seed = %d" % seed)
        np.random.seed(seed)
        random.seed(seed)

        if envname == 'cas':
            env = CasEnv(L=L, d=d, synthetic=synthetic, tabular=tabular, filename=filename)
        elif envname == 'pbm':
            beta = [1/(k+1) for k in range(10)]
            # beta = np.ones(10) # used in MovieLens part
            env = PbmEnv(L=L, d=d, beta=beta, synthetic=synthetic, tabular=tabular, filename=filename)

        for K in [10]:
            
            if tabular and envname == 'pbm':
                beta = [1/(k+1) for k in range(K)]
                env = PbmEnv(L=L, d=d, beta=beta, synthetic=synthetic, tabular=tabular, filename=filename)

            rrank = RecurRank(K, env, T)
            starttime = time.time()
            rregs = rrank.run()
            rruntime = time.time() - starttime
            if envname in ['cas', 'pbm']:
                np.savez(filename[:2]+'_'+envname+'_recur_K'+str(K)+'_'+str(seed), seed, np.ones(T)*env.get_best_reward(K)-rregs, rruntime)

            crank = CascadeLinUCB(K, env, T)
            starttime = time.time()
            cregs = crank.run()
            cruntime = time.time() - starttime
            if envname in ['cas', 'pbm']:
                np.savez(filename[:2]+'_'+envname+'_cas_K'+str(K)+'_'+str(seed), seed, np.ones(T)*env.get_best_reward(K)-cregs, cruntime)

            crank = CascadeLinTS(K, env, T)
            starttime = time.time()
            cregs = crank.run()
            cruntime = time.time() - starttime
            if envname in ['cas', 'pbm']:
                np.savez(filename[:2]+'_'+envname+'_casts_K'+str(K)+'_'+str(seed), seed, np.ones(T)*env.get_best_reward(K)-cregs, cruntime)

            trank = TopRank(K, env, T)
            starttime = time.time()
            tregs = trank.run()
            truntime = time.time() - starttime
            if envname in ['cas', 'pbm']:
                np.savez(filename[:2]+'_'+envname+'_top_K'+str(K)+'_'+str(seed), seed, np.ones(T)*env.get_best_reward(K)-tregs, truntime)



if __name__ == "__main__":
    # main(L=10000,d=5,T=500000,repeat=1,envname='cas',synthetic=True,tabular=False,filename='')
    main(L=10000,d=5,T=5000000,repeat=1,envname='pbm',synthetic=True,tabular=False,filename='')
    # main(L=1000,d=5,T=2000000,repeat=1,envname='pbm',synthetic=False,tabular=False,filename='ml_1100_1k.npy')

