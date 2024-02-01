# Problem

Solver engines fail to find optimal solutions, often with error infeasible, for real blockchain quantities.

There are several approaches to handles that:

- scale in amounts to fit some good range
- change model expression
- configure(including disable) presolvers
- tune solver configuration (including trying other solver)

## Scale in amounts to fit some good range

Here is proposed algorithm.

### Find sloppy oracle



cap big pools up to degree relative to tendered amount (we do not care tiny slippage)  - uses oracle ok
eliminate small pools relative to tendered amount(we do not care tiny routes up to some degree) - uses oracle ok
zoom all numbers to fit biggest pool into range
UPDATED: detect small number but important tokens pools (via oracle)
bump all such pools by factor until minimal  high price token pool in range(assuming big pool is in range too). So update here is that we do not bump using oracle, but we bump the token in all places using minimal factor possible to move token pools into range.
solve
if any bad still, backups (not coded):
8.1. use oracle to skew these assets, assuming it is ok for now as oracles are good
8.2. if we replace all such tiny amounts pools from uniswap to balancer (with weight) and replace tiny values with constrains with weights, we can assymetircally scale important tokens (skew). but if this is better needs many simulations to prove it is better at all and/or to have PhD optimization in math to tell if it is really better (until PhD will tell to run tons of simulations).  but I do not care. I use 8.1. But 8.2 is right question to answers by somebody later (edited) 
:black_heart:
1

23:34
that is for optimal solver engine approach