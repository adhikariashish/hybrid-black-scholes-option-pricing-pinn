# Derivation of the Geometric Brownian Motion Solution

We start with your GBM SDE (risk-neutral drift):

$$dS_t = r S_t\,dt + \sigma S_t\, dW_t$$

Approach: 
- We solve the problem in log-space because addition is easier,
then exponentiate to go back to price-space where multiplication lives,
and because randomness is involved, the correct tool is Itô’s lemma.


## Step 1: Rewrite in proportional form

Divide both sides by $S_t$ (since $S_t > 0$ for GBM):

$$\frac{dS_t}{S_t} = r\,dt + \sigma\, dW_t$$

If this were ordinary calculus, we'd now integrate and get $\log S_t$. But because $W_t$ is random, we must use Itô's lemma.

## Step 2: Apply Itô's lemma to $f(S_t) = \log S_t$

Let $X_t = S_t$. For $f(x) = \ln x$:
* $f'(x) = \frac{1}{x}$
* $f''(x) = -\frac{1}{x^2}$

Itô's lemma:

$$df(X_t) = f'(X_t)\,dX_t + \frac{1}{2} f''(X_t)\,(dX_t)^2$$

So:

$$d(\ln S_t) = \frac{1}{S_t} dS_t + \frac{1}{2}\left(-\frac{1}{S_t^2}\right) (dS_t)^2$$

## Step 3: Substitute $dS_t = rS_t dt + \sigma S_t dW_t$

First term:

$$\frac{1}{S_t} dS_t = \frac{1}{S_t}(rS_t dt + \sigma S_t dW_t) = r\,dt + \sigma\, dW_t$$

Now compute $(dS_t)^2$. Expand:

$$(dS_t)^2 = (rS_t dt + \sigma S_t dW_t)^2$$

Use the Itô rules (key identities):
* $(dt)^2 = 0$
* $dt\, dW_t = 0$
* $(dW_t)^2 = dt$

So only the $(\sigma S_t dW_t)^2$ survives:

$$(dS_t)^2 = \sigma^2 S_t^2 (dW_t)^2 = \sigma^2 S_t^2\, dt$$

Plug into the second term:

$$\frac{1}{2}\left(-\frac{1}{S_t^2}\right)(dS_t)^2 = \frac{1}{2}\left(-\frac{1}{S_t^2}\right)\sigma^2 S_t^2 dt = -\frac{1}{2}\sigma^2 dt$$

## Step 4: Combine terms

So:

$$d(\ln S_t) = (r\,dt + \sigma\,dW_t)\;-\;\frac{1}{2}\sigma^2 dt$$

$$\boxed{ d(\ln S_t) = \left(r - \frac{1}{2}\sigma^2\right) dt + \sigma\, dW_t }$$

This is  where the $-\frac{1}{2}\sigma^2$ term comes from.

## Step 5: Integrate from 0 to $T$

Integrate both sides:

$$\int_0^T d(\ln S_t) = \int_0^T \left(r - \frac{1}{2}\sigma^2\right) dt + \int_0^T \sigma\, dW_t$$

Left side:

$$\ln S_T - \ln S_0$$

First right integral:

$$\left(r - \frac{1}{2}\sigma^2\right)T$$

Second right integral:

$$\sigma (W_T - W_0) = \sigma W_T \quad \text{(since } W_0=0\text{)}$$

So:

$$\boxed{ \ln S_T = \ln S_0 + \left(r - \frac{1}{2}\sigma^2\right)T + \sigma W_T }$$

## Step 6: Exponentiate to solve for $S_T$

$$S_T = \exp(\ln S_T) = S_0 \exp\!\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma W_T\right)$$

$$\boxed{ S_T = S_0 \exp\!\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma W_T\right) }$$

## Step 7: Convert Brownian motion to a Normal random variable

A key fact:

$$W_T \sim \mathcal{N}(0, T)$$

So we can write:

$$W_T = \sqrt{T}\,Z,\quad Z\sim\mathcal{N}(0,1)$$

Final closed form:

$$\boxed{ S_T = S_0 \exp\!\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T}\, Z\right) }$$

## For simulation over many steps (what code uses)

For step size $\Delta t = T/n$:

$$\boxed{ S_{t+\Delta t} = S_t \exp\!\left(\left(r - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t}\, Z\right) }$$


# Example Calculation: GBM Closed-Form Solution

We'll use:
* $S_0 = 100$
* $r = 0.05$
* $\sigma = 0.20$
* $T = 1$ year
* pick one random draw: $Z = 0.30$ (just an example)

GBM closed-form:

$$S_T = S_0 \exp\!\left(\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right)$$

## Step 1) Compute $\tfrac{1}{2}\sigma^2$

$$\sigma^2 = 0.20^2 = 0.04$$

$$\tfrac{1}{2}\sigma^2 = 0.5 \times 0.04 = 0.02$$

## Step 2) Compute the drift term $(r - \tfrac{1}{2}\sigma^2)T$

$$r - \tfrac{1}{2}\sigma^2 = 0.05 - 0.02 = 0.03$$

$$(0.03)T = 0.03 \times 1 = 0.03$$

## Step 3) Compute the randomness term $\sigma\sqrt{T}\,Z$

$$\sqrt{T} = \sqrt{1} = 1$$

$$\sigma\sqrt{T}\,Z = 0.20 \times 1 \times 0.30 = 0.06$$

## Step 4) Add them to get the exponent

$$\text{exponent} = 0.03 + 0.06 = 0.09$$

## Step 5) Exponentiate and multiply by $S_0$

$$S_T = 100 \times e^{0.09}$$

Now compute $e^{0.09}$ (approx):
* $e^{0.09} \approx 1.094174$

So:

$$S_T \approx 100 \times 1.094174 = 109.4174$$

✅ **Final:**

$$\boxed{S_T \approx 109.42}$$

### What this shows

- Drift contributed +0.03 in the exponent (about +3% growth)

- Random shock contributed +0.06 (a positive “lucky” move)

- Combined ended up around +9.4% over the year for this one simulated world
