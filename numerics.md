# Problem

 Reserves and amounts can be big and/or small
 Thats leads to numeric infeasibility. Try to run two-assets.py with big numbers.
 We should scale in amounts and scale out.
 There are are two approaches:
 - find out reason of infeasibility and tune relevant constraints/parameter
 - scale in using raw limits, cap reservers against small tendered, and scaling in using oracle (including inner)
 This solution going second way.
 For more context this overview is nice https://www.youtube.com/watch?v=hYBAqcx0H18
 If solver fails, can try scale one or more times.
 Amid external vs internal oracle:
 - Internal oracle proves there is path at all
 - Internal oracle shows price for route, not price outside on some CEX, which is more correct
 - Internal oracle harder to attack/manipulate
 
 So using internal oracle is here.


Solver engines fail to find optimal solutions, often with error infeasible, for real blockchain quantities.

There are several approaches to handles that:

- scale in amounts to fit some good range
- change model expression
- configure(including disable) presolvers
- tune solver configuration (including trying other solver)

## Scale in amounts to fit some good range

Here is proposed algorithm.

### Find sloppy oracle

We can run forward only algorithm of depth N from start token to target token.

Oracle is built using inverse length, direct reservers weighting averaging across all venues.

So inner oracle accounts for price along possible routes.

Alternative is using external oracle not discussed here.  

### Cap big pools

So we know relative price on tendered asset to reservers.

Assuming that slippage (change in settlement price induced by settlements along the routes)

is neglectable if reserver much more bigger,

we just cap reserves in all big venues up to limit.

Here we okey with sloppy oracle, not precise with big mistake.

Assumes that pools disbalance in big pools is small.

### Eliminate small pools

If some pool is tiny fraction of tendered assets,

we remove that pools from search space.

Sloppy oracle is ok here too.

### Zoom

Zoom all numbers to fit biggest pool into range

### (Optional) Scale up important tokens

Detect small reservers in some venues.
 
Scale up these by some factor in all venues.

If largest reserver will not be upper than limit, retain rescale.

So we try to make some small numeric important tokens bigger.

As you may recall we did not eliminated such pools via oracle before.

### (Optional) Oracle rescale

In presence of high quality oracle we could scale all pools and scale on this.

Here is high risk to loose arbitrage.

### (Optional) Weight

We could replace some venues to be weighted, so can skew weight instead value of reserve.

### Solve

Run solver engine.

### Scale out

Return values to original scale.