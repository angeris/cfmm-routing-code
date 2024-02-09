# Problem

Here is more formal problem definition https://or.stackexchange.com/questions/11603/numerical-infeasibility-for-moving-numbers-along-some-specific-conversion-edges

Reserves numbers can be small and big, for example 10^6 and 10^21.

Inputs amounts can be as small and big too, for example as 10^3 and as big as 10^16

Price ratios can vary greatly. For example 10^12 of some token can be 10^3 of others.

Giving such big number ranges to solver leads to numeric infeasibility, engine fails and [slowness](https://www.youtube.com/watch?v=hYBAqcx0H18).

There are several approaches to handles that:

- scale in amounts to fit some good range
- change model expression
- configure(including disable) presolvers
- tune solver configuration (including trying other solver)

Also can run solver with different numeric fixes concurrently and see what works.

## Scale in amounts to fit some good range

Here is proposed algorithm for this approach.

**This algorithm better to run several times on range of inputs if purpose is to seek arbitrage**.

### Find sloppy inner oracle

We can run forward only algorithm of depth N from start token to any other token.

Oracle is built using inverse length, direct reserves weighting averaging across all venues reaching same token.

Assumes big pool are near optimal.

So inner oracle accounts for price along possible reachable routes.

Alternative is using external oracle not discussed here, because:  

- Internal oracle proves there is path at all
- Internal oracle shows price of real route, not price outside on some CEX, which is more correct in case of cross chain exchange
- Internal oracle harder to attack/manipulate

Here is formulation of oracle https://cs.stackexchange.com/questions/165350/optimization-of-value-over-network-flow-from-start-to-end-node-with-constant-fun?noredirect=1&lq=1

**Oracle is never used to find optimal solutions and values, but bad oracle may prevent find some best solutions.**

`oracalized value` - amount of some token expressed in price to tendered token, for tendered token it is `1.0`

`sloppy oracle` - oracle which mistakes price up to sum multiplier under 100x

### Cap big pools

So we know relative price of tendered asset to reserves and constant fees.

Assuming that slippage (change in settlement price induced by settlements along the routes)

is neglectable if reserves much more bigger,

we just cap reserves in all big venues up to limit.

Here we OK with sloppy oracle.

### Eliminate small pools

If some pool is tiny fraction of tendered assets,

we remove that pools from search space.

Sloppy oracle is ok here too.

### Zoom

Zoom all numbers to fit biggest pool of each each token into range.

#### Scale down big reserves tokens

If some numerically big reserves exists, scale down all reservers of that token to fit numeric limit.

#### Scale up these by some factor in all venues

Detect numerically small reserves in some venues.

Upscale all reserves of that token until it limits oracalized value.

So we try to make some small numeric important tokens bigger.

As you may recall we did not eliminated such pools via oracle before.

### (Optional) Oracle rescale

In presence of high quality oracle we could scale all pools and scale on this.

Here is high risk to loose arbitrage(optimality).

### (Optional) Weight

We could replace some venues to be weighted(Balancer whitepaper), so can skew weight instead value of reserve.

### Solve

Run solver engine.

### Scale out

Return values to original scale.