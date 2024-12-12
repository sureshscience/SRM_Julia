### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ c8590734-c738-4225-9e4e-372f5ef7a63c
using Plots

# ╔═╡ bae4a3e4-2224-4e5d-ab15-bbdf86310ba8
begin
	using PlutoUI
	PlutoUI.TableOfContents(title = "Contents")
end

# ╔═╡ 133172e0-ee0f-11eb-3d88-efecd8115ca7
md"""
# Modelling infectious diseases
"""

# ╔═╡ 0f33ac97-eb86-40cf-aa65-8ffb51cdf37c
md"""
## Objectives
- Create and compare two different ``SIR`` models for infectious disease
  - Mathematically, the classical ``SIR`` model given in terms of differential equations
  - Computationally, an ``SIR`` model considering proximity of individuals
- _**Consider methods of increasing efficiency in Julia code**_
"""

# ╔═╡ 70d33a7a-edac-421c-a6e0-7acfd523871e
md"""
## What is an SIR model?

An ``SIR`` model models an infectious disease by splitting the population into three categories
- **Susceptible** (``S``) - Those who are susceptible to the disease, having not come into contact with it yet
- **Infectious** (``I``) - Those who have the disease and are able to pass it on to the susceptible population
- **Recovered** (``R``) - Those who have had the disease in the past but can no longer pass it on. This category is sometimes called **Removed** to be more inclusive to those who died from the disease

The model then has rules for how the size of the three categories changes with time. In the models in this case study, the total population `` N = S + I + R `` is fixed, but this doesn't have to be the case.

The two models that I will create in the following studies are:
- A simple yet effective model based on an intuitive system of differential equations, which I will refer to as the [mathematical model](#ed6cab5f-4fdb-4680-9978-6024aa57595c)
- A more complicated model attempting to simulate the disease passing between individuals, which I will refer to as the [computational model](#99d6153f-38ae-4ba4-a96e-509e9094bdb1)
"""

# ╔═╡ ed6cab5f-4fdb-4680-9978-6024aa57595c
md"""
## Mathematical model

The most simple and elegant ``SIR`` model is given by three differential equations:

```math
\frac{dS}{dt} = - \beta I S
```
```math
\frac{dI}{dt} = \beta I S - \nu I
```
```math
\frac{dR}{dt} = \nu I
```

although since we know that `` S + I + R `` is fixed, and the equations do not depend directly on ``R``, the first two will suffice. Here, ``\beta`` is the rate of infection, and ``\nu`` is the rate of recovery, which are parameters that I will be able to change later. 

To implement this in Julia, I will use the Forward Euler algorithm where

```math
\frac{dF}{dt} = f(t), \qquad F(0) = F_0
```

can be approximated for `` t > 0 `` by iteration of the form

```math
F(0) = F_0, \qquad F((n+1)\delta{}t) \approx F(n\delta{}t) + f(n\delta{}t) \delta{}t
```

where ``\delta{}t`` is some chosen step size (smaller is more accurate but requires more computation).

I start by choosing `δt` and a final time `T` up to which I will run the simulation.
"""

# ╔═╡ 728e28a5-d807-4751-858c-4a94873463be
δt = 0.001

# ╔═╡ 2598752b-f99f-457b-8192-b9e256c72e1e
T = 1

# ╔═╡ 828ec2df-aa76-44ff-b951-880e159cb82f
times = 0:δt:T

# ╔═╡ 262d57fa-8ab8-4a9e-a826-3c8550692811
ntimes = length(times)

# ╔═╡ fb7bb03b-bcf3-4d2c-a066-294256e2d92e
md"""
I also choose a population size `N`. The subscript `₁` denotes that this is the first model of the two.
"""

# ╔═╡ dc26b16a-25ba-4b16-9b52-16b912c4b8bb
N₁ = 10000

# ╔═╡ b75775ab-a852-4571-8e59-63b5083db3c2
md"""
I initialise vectors the same size as the list of times to track ``S`` and ``I`` (as mentioned earlier there is no need to track ``R``), with the initial values being one infectious individual and the rest susceptible.
"""

# ╔═╡ 5363d193-fb8a-469d-9e87-09ece19a01fe
S₁ = zeros(size(times));

# ╔═╡ 1751daa5-78e7-471c-8576-1c9acc2dd68c
S₁[1] = N₁ - 1;

# ╔═╡ 0533a017-889b-4f72-ad21-58f762f4fae2
I₁ = zeros(size(times));

# ╔═╡ 547bc8b9-22fb-4143-9eb6-8f2b4090ed65
I₁[1] = 1;

# ╔═╡ 64f43f00-eacb-4039-81b0-d09b94af8923
md"""
Finally, I choose values for the two parameters `β` and `ν`.
"""

# ╔═╡ 3fa04a1c-25eb-4fd4-aa77-f0296c83dd31
β = 0.005

# ╔═╡ 688bebd2-2d88-441c-90f6-9597de92a176
ν = 4

# ╔═╡ 310433cf-f3a6-4e4a-8f2c-c5d5967b5ad4
md"""
Forward Euler iteration can now begin. I am careful to make sure that `S`, `I`, and `S + I` remain between `0` and `N` at all times, although the model has clearly failed if this ever becomes a problem.
"""

# ╔═╡ 907983f7-0950-49ae-ba41-85da7ac577ed
for i ∈ 2:ntimes
    δS₁ = - β * I₁[i-1] * S₁[i-1] * δt
    δI₁ = - δS₁ - ν * I₁[i-1] * δt
    S₁[i] = min(max(S₁[i-1] + δS₁, 0),N₁)
    I₁[i] = min(max(I₁[i-1] + δI₁, 0),N₁ - S₁[i])
end

# ╔═╡ 16ff8546-92f0-4578-9310-28d8db1cb6e2
md"""
I can now calculate `R` from the values obtained for `S` and `I`.
"""

# ╔═╡ e20a2623-e0bc-4421-8726-8f81554d63b1
R₁ = N₁ .- (S₁ + I₁);

# ╔═╡ 1753a908-a344-40f6-b25d-6dbc124f5272
md"""
The `Plots` package allows me to visualise the results, which I will do in two forms, a line graph, and a stacked area graph.
"""

# ╔═╡ 5bb45fc2-1361-4555-952d-d50f9e6aea77
lineplot₁ = plot(
	times,
	[S₁ I₁ N₁ .- (S₁ + I₁)],
	label = ["Susceptible" "Infectious" "Recovered"],
	color = [:gold :red :blue]
)

# ╔═╡ 7fe01089-0b35-417e-b857-1b0ba3653ccd
areaplot₁ = areaplot(
	times,
	[S₁ I₁ N₁ .- (S₁ + I₁)],
	label = ["Susceptible" "Infectious" "Recovered"],
	color = [:gold :red :blue]
)

# ╔═╡ e794f4a0-fcad-427e-af49-7fd20ed11a10
md"""
An interesting alteration to make to this model is to introduce some randomness, for which I will need to introduce a vector to track ``R``. The randomness I will create by multiplying `δS`, `δI`, and `δR` by log-normally distributed random variables of parameters `μ = 0`, and `σ` a parameter I can set (note that `σ = 0` is equivalent to no randomness), and then normalising to keep `S + I + R` fixed.

It is now particularly essential that `S`, `I`, and `R` are kept between `0` and `N` since the randomness increases the chance for them to end up outside of that range.
"""

# ╔═╡ b3e54a43-a997-43bf-994c-19100d49f765
S₁′ = zeros(size(times));

# ╔═╡ f33ac3be-782d-4183-9c84-ee9c161d33ee
S₁′[1] = N₁ - 1;

# ╔═╡ 3b63e289-dddf-4460-bde9-ce6acd7607ab
I₁′ = zeros(size(times));

# ╔═╡ 6b7a449f-3fcc-457d-b268-39fddda134a2
I₁′[1] = 1;

# ╔═╡ 8c207e61-241c-4af1-a1dd-0f5580157de8
R₁′ = zeros(size(times));

# ╔═╡ 61c09dcb-b03a-485d-a954-9ebbc9501d35
σ = 1

# ╔═╡ 1641868a-a25b-499d-bc72-e6fff52036bc
for i ∈ 2:ntimes
    δS₁′ = - β * I₁′[i-1] * S₁′[i-1] * δt
	δR₁′ = ν * I₁′[i-1] * δt
	δI₁′ = - δS₁′ - δR₁′
    S₁′[i] = max(S₁′[i-1] + δS₁′ * exp(σ * randn()), 0)
    I₁′[i] = max(I₁′[i-1] + δI₁′ * exp(σ * randn()), 0)
	R₁′[i] = max(R₁′[i-1] + δR₁′ * exp(σ * randn()), 0)
	scale = (S₁′[i] + I₁′[i] + R₁′[i])/N₁
	S₁′[i] /= scale; I₁′[i] /= scale; R₁′[i] /= scale
end

# ╔═╡ 4670197e-4c4e-49ec-ad7d-ffdd0701631e
md"""
Plotting `S`, `I` and `R` now gives:
"""

# ╔═╡ d2821018-da14-45c2-a98e-47edf882a867
plot(
	times,
	[S₁′ I₁′ R₁′],
	label = ["Susceptible" "Infectious" "Recovered"],
	color = [:gold :red :blue]
)

# ╔═╡ a52bbb3c-3f25-46a0-819c-dce56bd7c75c
areaplot(
	times,
	[S₁′ I₁′ R₁′],
	label = ["Susceptible" "Infectious" "Recovered"],
	color = [:gold :red :blue]
)

# ╔═╡ 7a039123-2afd-4f2d-b47b-d05f3cb4f13c
md"""
There is only a small effect of randomness on making the curves less smooth. What it does do, however, is dramatically quicken both processes of infection and recovery. This can be explained as the exponential nature of the processes causing a feedback loop which amplifies the effects of random increases in rate while dulling the effects of random decreases in rate.
"""

# ╔═╡ 99d6153f-38ae-4ba4-a96e-509e9094bdb1
md"""
## Computational model
The second model which I will use simulates a population of ``N`` people with the disease propagating only between neighbours:
- ``N`` people are arranged in a grid, with one starting out as infectious and the rest susceptible
- At each step:
  - Any infectious individual passes on the disease to any of their four neighbours independently with probability ``p``
  - Any infectious individual recovers with probability ``q`` (can happen on the same step as passing on the disease, but not in the same step as getting passed the disease)
- This is run for a predetermined number of steps, with the numbers ``S``, ``I``, and ``R`` kept track of at all times

This model has some features that the first does not which may make it more realistic, such as:
- The population is discrete, allowing the disease to die out more easily
- The disease is localised so cannot infect those who are not in contact with it

However, due to the simulation of individuals rather than the simulation of the population as a whole, this is inevitably more computationally intensive. Hence, efficiency will be essential to make this model usable.

### Setting up the model
To begin with, I will set up the parameters for the model.
"""

# ╔═╡ df4d71fd-a752-46fd-bb18-908a23ee0cd0
sqrtN₂ = 100

# ╔═╡ 1086d559-ef39-405a-a113-a060b391ad9e
N₂ = sqrtN₂^2

# ╔═╡ 25922ea7-4054-4054-ba4a-f93dbee85503
p = 0.4

# ╔═╡ b39fb04a-1074-4441-9d6d-3612674475d9
q = 0.01

# ╔═╡ a87ed703-f514-41cf-8262-b5ace0a28e45
maxsteps = 500

# ╔═╡ b24a796c-4908-4ff9-b392-0781574829e4
md"""
The population of size `N₂` will be stored as a matrix of numbers (`sqrtN₂` by `sqrtN₂`), where `1` represents susceptible, `2` represents infectious, and `3` represents recovered.
"""

# ╔═╡ 7709c287-c7b8-4ffe-a257-4ada27454db3
population = fill(1, (sqrtN₂, sqrtN₂));

# ╔═╡ fdc5dd62-50e7-4423-880b-1dbde1aebea3
population[rand(1:N₂)] = 2;

# ╔═╡ a565879c-2938-4f5b-8de8-9807cdff3948
md"""
As before, vectors `S₂` and `I₂` will keep track of the number of susceptible and infectious individuals respectively after each step.
"""

# ╔═╡ d74fe014-47a1-4636-9360-83fc311307d2
S₂ = zeros(maxsteps+1);

# ╔═╡ b8903866-81e9-42b0-8bd7-263d5cbaf2b4
S₂[1] = N₂ - 1;

# ╔═╡ 94272156-621b-4ed5-8582-c079625fa977
I₂ = zeros(maxsteps+1);

# ╔═╡ ef4051e0-38f2-4dc6-a765-b0c76a4b0705
I₂[1] = 1;

# ╔═╡ 8fbf94ce-50f5-48a6-91ce-d2f67fa505e4
md"""
As an additional form of output, I will create an animation of the population at each stage, using the `Plots` package. I use the function `populationplot` to convert the matrix `population` into a heatmap where `1` is yellow, `2` is red, and `3` is blue (which is why I use numbers in the matrix). In order to maintain these colours, three additional pixels of each of these values are added, as otherwise `Plots` will change the scale if the range of values of the plot is not 1 to 3.
"""

# ╔═╡ dfc867ac-d8d7-4f90-9410-03746e134443
function populationplot(population::Matrix{Int64})
    return heatmap(
        hcat(population, fill(missing, sqrtN₂),
			[1,2,3, fill(missing, sqrtN₂ - 3)...]),
        legend = false,
        color = [:gold, :red, :blue],
        size = (4*sqrtN₂,4*sqrtN₂),
        showaxis = false,
        ticks = false
    )
end

# ╔═╡ f4e7d095-13c3-40dc-b248-61d8a065d61c
begin
	anim₂ = Animation()
	frame(anim₂, populationplot(population))
end;

# ╔═╡ 559aeb30-0d2e-4077-9af1-b4ed4702f9e5
md"""
I am now ready to carry out the iteration. For each step, I would like to do the following:

- Initialise empty lists of indices for points which will become red and points that will become blue
```julia
reds = CartesianIndex{2}[]
blues = CartesianIndex{2}[]
```

- Iterating over each member of the population, check whether they are infectious
```julia
for j ∈ 1:sqrtN₂, i ∈ 1:sqrtN₂
    if population[i,j] == 2
		# loop
	end
end
```

- For those which are infectious, infect each susceptible neighbour with independent probability `p`, and recover with probability `q`, adding the appropriate indices to the lists `reds` and `blues`
```julia
i > 1 && population[i-1,j] == 1 && rand() < p && push!(reds,CartesianIndex(i-1,j))
j > 1 && population[i,j-1] == 1 && rand() < p && push!(reds,CartesianIndex(i,j-1))
i < sqrtN₂ && population[i+1,j] == 1 && rand() < p &&
	push!(reds,CartesianIndex(i+1,j))
j < sqrtN₂ && population[i,j+1] == 1 && rand() < p &&
	push!(reds,CartesianIndex(i,j+1))
rand() < q && push!(blues,CartesianIndex(i,j))
```

- Once this is complete, update the status of the population with the new `reds` and `blues`
```julia
population[reds]  .= 2
population[blues] .= 3
```

- Create the next frame of the animation
```julia
frame(anim₂, populationplot(population))
```

- Calculate the new number of susceptible and infectious
```julia
nnewreds, nnewblues = (length ∘ unique! ∘ sort!)(reds), length(blues)
S₂[n] = S₂[n-1] - nnewreds
I₂[n] = I₂[n-1] + nnewreds - nnewblues
```

- If the disease has died out (i.e. no more infectious), stop the loop, since there is no point in continuing
```julia
I₂[n] == 0 && (S₂ = S₂[1:n]; I₂ = I₂[1:n]; break)
```

This I combine into one large loop.
"""

# ╔═╡ dc815f4e-a731-4e6d-987d-9a9db7ee5270
for n ∈ 2:(maxsteps+1)
    reds = CartesianIndex{2}[]
    blues = CartesianIndex{2}[]
    for j ∈ 1:sqrtN₂, i ∈ 1:sqrtN₂
        if population[i,j] == 2
            i > 1      && population[i-1,j] == 1 && rand() < p &&
				push!(reds,CartesianIndex(i-1,j))
            j > 1      && population[i,j-1] == 1 && rand() < p &&
				push!(reds,CartesianIndex(i,j-1))
            i < sqrtN₂ && population[i+1,j] == 1 && rand() < p &&
				push!(reds,CartesianIndex(i+1,j))
            j < sqrtN₂ && population[i,j+1] == 1 && rand() < p &&
				push!(reds,CartesianIndex(i,j+1))
            rand() < q && push!(blues,CartesianIndex(i,j))
        end
    end

    population[reds]  .= 2
    population[blues] .= 3
    frame(anim₂, populationplot(population))

    nnewreds, nnewblues = (length ∘ unique! ∘ sort!)(reds), length(blues)
    S₂[n] = S₂[n-1] - nnewreds
    I₂[n] = I₂[n-1] + nnewreds - nnewblues
    I₂[n] == 0 && (S₂ = S₂[1:n]; I₂ = I₂[1:n]; break)
end

# ╔═╡ 3e2119c2-9482-482b-8caf-f9d386ae0522
md"""
From `S` and `I`, I calculate `R` as before.
"""

# ╔═╡ 466ab0b6-fb5e-4354-ae91-20c7bc20a583
R₂ = N₂ .- (S₂ + I₂);

# ╔═╡ 8ef0ac34-d033-4515-af2d-9f44c6749977
md"""
Finally, the visualisations can be created. First, I plot the animation of the spread of the infection, and then the same two graphs as were plotted for the first model.
"""

# ╔═╡ ba404ab9-794d-499f-b41a-7601a994e42e
 gif₂ = Plots.gif(anim₂)

# ╔═╡ fa9bcbb8-927f-45cd-96c1-14ff760b2af8
lineplot₂ = plot(
	0:length(S₂)-1,
	[S₂ I₂ R₂],
	label = ["Susceptible" "Infectious" "Recovered"],
	color = [:gold :red :blue]
)

# ╔═╡ fff03d6e-741b-48a6-b90f-89a52f4d707e
areaplot₂ = areaplot(
	0:length(S₂)-1,
	[S₂ I₂ R₂],
	label = ["Susceptible" "Infectious" "Recovered"],
	color = [:gold :red :blue]
)

# ╔═╡ df111c85-33bf-4ca1-846e-a20e3491dcbd
md"""
### Efficiency considerations

The second model took noticeably more effort for me to program than the first. This is mainly because it is more complicated, and in this instance, with complexity comes slowness. For this reason, I had to make many changes to my initial attempt in order to optimise it.

Here is a snippet of an early draft of the program for the second model:
```julia
population = fill(:gold, (sqrtN₂, sqrtN₂))
population[rand(1:N₂)] = :red

anim₂ = Animation()
function populationplot(population::Matrix{Symbol})
	return scatter(
		[(i,j) for j ∈ 1:sqrtN₂, i ∈ 1:sqrtN₂][:],
		markersize = 3,
		markercolor = population[:],
		markerstrokewidth = 0,
		size = (4*sqrtN₂, 4*sqrtN₂),
		legend = false,
		showaxis = false,
		ticks = false
	)
end
frame(anim₂,populationplot(population))

for n ∈ 2:(maxsteps+1)
	newpopulation = copy(population)
	for j ∈ 1:sqrtN₂, i ∈ 1:sqrtN₂
        if population[i,j] == :red
            i > 1 && population[i-1,j] == :gold && rand() < p &&
				newpopulation[i-1,j] = :red
            j > 1 && population[i,j-1] == :gold && rand() < p &&
				newpopulation[i,j-1] = :red
            i < sqrtN₂ && population[i+1,j] == :gold && rand() < p &&
				newpopulation[i+1,j] = :red
            j < sqrtN₂ && population[i,j+1] == :gold && rand() < p &&
				newpopulation[i,j+1] = :red
            rand() < q && newpopulation[i,j] = :blue
        end
    end
	
	population = newpopulation
	frame(anim₂,populationplot(population))
	
	S₂[n] = count(==(:gold), population)
	I₂[n] = count(==(:red), population)
	I₂[n] == 0 && (S₂ = S₂[1:n]; I₂ = I₂[1:n]; break)
end
```
"""

# ╔═╡ 9756b3f3-e826-4b6d-ac4f-54ea7a5d43d1
md"""
With experience coding in Julia (and in general), you can pick up tips and tricks for efficiency that become almost automatic for you to include. Some examples of this in this instance are already included in my early draft, and some could be added in:

- Julia orders the elements of matrices by columns then rows, i.e the next element of a matrix `A` after `A[i,j]` is `A[i+1,j]` (the element below it). Note that this is opposite to some other languages, such as Python. A consqeuence of this is that when looping over matrices, the outer loops should loop over the columns, and the inner loop over the rows, since then the entries are being accessed in exactly the order that they lie in memory. In the example above, this is the reason why I have written:
```julia
for j ∈ 1:sqrtN₂, i ∈ 1:sqrtN₂
```
- Another way of fixing this is to use `eachindex`, which gives an efficient way of iterating over an array (with syntax `for i ∈ eachindex(A)`). This is better when the row and column indices are irrelevant to the operation inside the loop, but I needed the indices, it isn't the right choice here.

- If multiple conditions need to be checked, it makes sense to check the fastest and/or most likely to fail first, since then the loop can move on quicker. This is seen in the draft, where the probability `p` checks are only made after checking that the target individual is susceptible, since that is a quick operation and is quite likely to not be true

- There is no need to work with large amounts of data when a small amount will do, in particular the matrix `newpopulation`, which is inefficiently copied from `population` only for most of the entries to remain the same since no infection or recovery happened there. This I fixed by introducing the lists `reds` and `blues` of points which have become red/blue, and then changing the values in the matrix `population` once finished, as can be seen in the final program. The `CartesianIndex` type is used to store these indices since it allows the assignment of values to a list of indices with `.=` as shown below, while storing indices as a vector of tuples or as a matrix would not give as neat a solution.
```julia
reds = CartesianIndex{2}[]
blues = CartesianIndex{2}[]
⋮
population[reds]  .= :red
population[blues] .= :blue
```

- Since I now have the list of `reds` and `blues` that are changing at this step, there is no need to count `S[n]` and `I[n]` so inefficiently. Instead, I need only count how many are changing and add to / subtract from the previous value. For `blues`, this is simple, but for `reds` there could be duplicates (if two infectious people infect the same susceptible person that they both neighbour). `unique!` would do this, but the documentation notes that if the order of the original points isn't needed (which it isn't), then `unique! ∘ sort!` is better.
```julia
nnewreds, nnewblues = (length ∘ unique! ∘ sort!)(reds), length(blues)
S₂[n] = S₂[n-1] - nnewreds
I₂[n] = I₂[n-1] + nnewreds - nnewblues
```

- More examples of this sort of optimisation can be found at <https://docs.julialang.org/en/v1/manual/performance-tips/>
"""

# ╔═╡ 6733d7f2-c434-4719-9ae8-e15fa58779ce
md"""
The next place to look for improvements in efficiency is in profiling, which helps to find which places in the code are taking the longest, and where improvements to be made. This is implemented by the inbuilt package `Profile`, although for better visualisation, the package `ProfileView` is needed.
```julia
using ProfileView
```
Profiling works best for testing individual functions, although it is important to make sure that the functions are compiled first before running the test (this can be achieved by running the function, or by profiling twice and taking the second result as true). However, in this instance, I don't have a single function to test, but an entire program, so I run the program once to compile it, and then run
```julia
ProfileView.@profview include("<file-path>")
```
This produces a flame graph, which is a graph made up of stacked horizontal bars. Each bar represents a function called within the process of running the program, with the length denoting the length of time spent in that function, and each bar lying below all the bars representing the functions that it calls within that time. Usually, the best place to look for potential improvements are the longest bars, constituting the functions that are the slowest / called the most times within the program. Once you have identified these places, you can consider how you can rewrite these sections of code to speed them up. Flame graphs can also be saved to compare different iterations of the program, which can also help this process.

An advantage of using editors is that they can also have their own inbuilt tools for profiling. One example of this is in VSCode, where the Julia extension comes with an inbuilt package `VSCodeServer` with its own version of `@profview`. Again, run the function or program once, before running
```julia
VSCodeServer.@profview include("<file-path>")
```
Instead of a flame graph, this produces a list of all of the functions called in the process of running ordered by "Self Time", that is the time for the functions themselves to run, ignoring those functions that they call. It also annotates the code with times, showing the lines of code which take the most. The flame graph from `ProfileView` can be obtained from the same run with
```julia
ProfileView.view()
```
Both methods of visualisation of the profile are valuable to understand where to look for gains in efficiency.

In the case of my program, I found that the majority of the time came from calls to `Plots` functions (and related packages, such as `GR` which is the backend that I am using for generating the graphs). My initial thought was that this may be because drawing the entire graph after each step may be inefficient, but amending this to instead add red and blue points to the graph at each step made little difference. This seems to be because with so many points on the graph, drawing it in order to capture a frame gets slower and slower with each step.

The solution that I settled on was to shift from manually drawing a scatter graph from a matrix of colours to drawing a heatmap from a matrix of numbers with custom colours. The heatmap is plotted similarly to the scatter graph, however it is already optimised to be efficient, unlike my own code, so it is a better choice.
"""

# ╔═╡ 53984f21-33d9-46b8-b08c-d15e044f2752
md"""
If the program still isn't fast enough, it is possible that more radical changes need to be made. The algorithm itself may be at fault, in which case there is nothing more that you can do then attempt to alter it or look elsewhere for a better one (if the constraints of the project allow you to do so). An important (but by no means only) consideration to make is the complexity of the algorithm, that is the rate at which the speed of the algorithm scales with increasing the size of the inputs or relevant parameters. Lower complexity algorithms tend to be faster, although if a large amount of overhead is required to lower the complexity it may not be worthwhile for realistically sized inputs.

For the mathematical model, the most relevant parameter is `ntimes` (or `δt` assuming that `T` is fixed). Each loop contains the exact same calculations, so the total complexity of the algorithm is ``O(n)`` (where ``n = `` `ntimes`).

For the computational model, (ignoring early termination, since complexity calculations always assume the worst case) the length of the loop is `maxsteps`, but the time that each iteration takes depends on the size of `I`. The worst case is that all neighbours are infected and none ever recover, with the assumption that `N` is large enough that this doesn't hit the border within `maxsteps` iterations, giving a sequence
```math
I_n = 1, 5, 13, 25,...
```
which are the [centered square numbers](http://oeis.org/A001844), increasing at a rate of ``I_n = O(n^2)``. Hence, the complexity of the second algorithm is (in terms of ``m = `` `maxsteps`) is
```math
\sum_{n = 2}^{m+1} O(n^2) = O(m^3)
```

This, however, demonstrates the limitations of considering complexity, since I already know that this part of the algorithm isn't the slow bit (even if theoretically it may be for larger `maxsteps`). In this instance, the higher complexity is pretty irrelevant for explaining the discrepancy in speed between the two algorithms, instead it is the formation of the animated visualisation which takes up most of the time.
"""

# ╔═╡ 23608a2c-77cb-4401-b598-9a7eb3d8d469
md"""
## Comparison of the two models

Since we have line graphs and stacked area graphs for both models, we can examine them side-by-side to compare the two models. On the left are the results of the first model (not including randomness), and on the right are the results of the second:

*Note for Pluto users: the analysis below is performed using the default parameters. If you have experimented with different values, it may be best to reload the notebook to see the intended graphs*
"""

# ╔═╡ 3df2b0f4-a7ae-4ad6-aad5-255330edf54b
plot(lineplot₁, areaplot₁, lineplot₂, areaplot₂, layout = (2,2), size = (800,600))

# ╔═╡ 87cac64d-90a8-431c-8179-7f7462e5c11a
md"""
Obviously, the two models are parameterised completely differently, so there is no reason to expect them to match perfectly. However, there is one clear difference between the two, which is the rate of propagation of the infection through the population, which is markedly quicker in the mathematical model.

To explain the difference, we need only look at the two algorithms. From the differential equations defining the first, when ``I`` is small and ``S`` is large,
`` \frac{dI}{dt} \approx \beta I S ``, giving (near) exponential growth. Meanwhile, as already seen when considering the complexity of the second algorithm, the growth in ``I`` is at most quadratic, and indeed with this parametrisation, is almost linear. Hence, even though the computational model is intended to be more realistic, it cannot model the phenomenon of exponential growth of infections which is seen in the real world. In my view, this is probably because although some level of contact is simulated, it does not come close to the sort of contact between individuals which actually occurs, since individuals only come into contact with four people all of whom are already in close contact anyway (perhaps it is more accurate as a model of maths students than the population as a whole).

In contrast, the recovery curves for the two models are almost identical. Again, this can be predicted by studying the algorithms. In the first, with `` S \approx 0 ``, `` \frac{dI}{dt} \approx - \nu I ``, which results in exponential decay. For the second model, each infectious person recovers with probability `q`, so ``I_n \approx q I_{n-1} ``, again giving exponential decay. The parameters that I have chosen make this even more obvious since they cause the blue Recovered curves to resemble each other very well, indeed:
"""

# ╔═╡ ba1205e6-dcde-4c44-a246-5bc180085ec7
begin
	recoverycomparison = plot(
		times,
		R₁,
		color = :deepskyblue,
		label = "Mathematical model",
		legend = :topleft,
		showaxis = false,
		ticks = false
	)
	plot!(
		recoverycomparison,
		(0:maxsteps) / maxsteps,
		R₂,
		color = :navy,
		label = "Computational model"
	)
end

# ╔═╡ 89c2bc30-220c-4add-8bc8-21b5b610d45b
md"""
Overall, I think that the mathematical model is a better model for infectious disease, although with some alterations to increase contact, the computational model may be able to equal it, with the added bonus of a visualisation of the disease spreading among the population rather than just the macroscopic graphs that the its competitor provides.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Plots = "~1.22.1"
PlutoUI = "~0.7.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "88aeb90ba893017b74f4a56141995b7481654a0f"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "13951eb68769ad1cd460cdb2e64e5e95f1bf123d"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.Extents]]
git-tree-sha1 = "81023caa0021a41712685887db1fc03db26f41f5"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.4"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "d189c6d2004f63fd3c91748c458b09f26de0efaa"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.61.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "59107c179a586f0fe667024c5eb7033e81333271"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.2"

[[deps.GeoInterface]]
deps = ["Extents", "GeoFormatTypes"]
git-tree-sha1 = "2f6fce56cdb8373637a6614e14a5768a88450de2"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.7"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ad31332567b189f508a3ea8957a2640b1147ab00"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+1"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "650a022b2ce86c7dcfbdecf00f78afeeb20e5655"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.2"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "e7523dd03eb3aaac09f743c23c1a553a8c834416"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "1165b0443d0eca63ac1e32b8c0eb69ed2f4f8127"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─133172e0-ee0f-11eb-3d88-efecd8115ca7
# ╟─0f33ac97-eb86-40cf-aa65-8ffb51cdf37c
# ╟─70d33a7a-edac-421c-a6e0-7acfd523871e
# ╟─ed6cab5f-4fdb-4680-9978-6024aa57595c
# ╠═728e28a5-d807-4751-858c-4a94873463be
# ╠═2598752b-f99f-457b-8192-b9e256c72e1e
# ╠═828ec2df-aa76-44ff-b951-880e159cb82f
# ╠═262d57fa-8ab8-4a9e-a826-3c8550692811
# ╟─fb7bb03b-bcf3-4d2c-a066-294256e2d92e
# ╠═dc26b16a-25ba-4b16-9b52-16b912c4b8bb
# ╟─b75775ab-a852-4571-8e59-63b5083db3c2
# ╠═5363d193-fb8a-469d-9e87-09ece19a01fe
# ╠═1751daa5-78e7-471c-8576-1c9acc2dd68c
# ╠═0533a017-889b-4f72-ad21-58f762f4fae2
# ╠═547bc8b9-22fb-4143-9eb6-8f2b4090ed65
# ╟─64f43f00-eacb-4039-81b0-d09b94af8923
# ╠═3fa04a1c-25eb-4fd4-aa77-f0296c83dd31
# ╠═688bebd2-2d88-441c-90f6-9597de92a176
# ╟─310433cf-f3a6-4e4a-8f2c-c5d5967b5ad4
# ╠═907983f7-0950-49ae-ba41-85da7ac577ed
# ╟─16ff8546-92f0-4578-9310-28d8db1cb6e2
# ╠═e20a2623-e0bc-4421-8726-8f81554d63b1
# ╟─1753a908-a344-40f6-b25d-6dbc124f5272
# ╠═c8590734-c738-4225-9e4e-372f5ef7a63c
# ╠═5bb45fc2-1361-4555-952d-d50f9e6aea77
# ╠═7fe01089-0b35-417e-b857-1b0ba3653ccd
# ╟─e794f4a0-fcad-427e-af49-7fd20ed11a10
# ╠═b3e54a43-a997-43bf-994c-19100d49f765
# ╠═f33ac3be-782d-4183-9c84-ee9c161d33ee
# ╠═3b63e289-dddf-4460-bde9-ce6acd7607ab
# ╠═6b7a449f-3fcc-457d-b268-39fddda134a2
# ╠═8c207e61-241c-4af1-a1dd-0f5580157de8
# ╠═61c09dcb-b03a-485d-a954-9ebbc9501d35
# ╠═1641868a-a25b-499d-bc72-e6fff52036bc
# ╟─4670197e-4c4e-49ec-ad7d-ffdd0701631e
# ╠═d2821018-da14-45c2-a98e-47edf882a867
# ╠═a52bbb3c-3f25-46a0-819c-dce56bd7c75c
# ╟─7a039123-2afd-4f2d-b47b-d05f3cb4f13c
# ╟─99d6153f-38ae-4ba4-a96e-509e9094bdb1
# ╠═df4d71fd-a752-46fd-bb18-908a23ee0cd0
# ╠═1086d559-ef39-405a-a113-a060b391ad9e
# ╠═25922ea7-4054-4054-ba4a-f93dbee85503
# ╠═b39fb04a-1074-4441-9d6d-3612674475d9
# ╠═a87ed703-f514-41cf-8262-b5ace0a28e45
# ╟─b24a796c-4908-4ff9-b392-0781574829e4
# ╠═7709c287-c7b8-4ffe-a257-4ada27454db3
# ╠═fdc5dd62-50e7-4423-880b-1dbde1aebea3
# ╟─a565879c-2938-4f5b-8de8-9807cdff3948
# ╠═d74fe014-47a1-4636-9360-83fc311307d2
# ╠═b8903866-81e9-42b0-8bd7-263d5cbaf2b4
# ╠═94272156-621b-4ed5-8582-c079625fa977
# ╠═ef4051e0-38f2-4dc6-a765-b0c76a4b0705
# ╟─8fbf94ce-50f5-48a6-91ce-d2f67fa505e4
# ╠═dfc867ac-d8d7-4f90-9410-03746e134443
# ╠═f4e7d095-13c3-40dc-b248-61d8a065d61c
# ╟─559aeb30-0d2e-4077-9af1-b4ed4702f9e5
# ╠═dc815f4e-a731-4e6d-987d-9a9db7ee5270
# ╟─3e2119c2-9482-482b-8caf-f9d386ae0522
# ╠═466ab0b6-fb5e-4354-ae91-20c7bc20a583
# ╟─8ef0ac34-d033-4515-af2d-9f44c6749977
# ╠═ba404ab9-794d-499f-b41a-7601a994e42e
# ╠═fa9bcbb8-927f-45cd-96c1-14ff760b2af8
# ╠═fff03d6e-741b-48a6-b90f-89a52f4d707e
# ╟─df111c85-33bf-4ca1-846e-a20e3491dcbd
# ╟─9756b3f3-e826-4b6d-ac4f-54ea7a5d43d1
# ╟─6733d7f2-c434-4719-9ae8-e15fa58779ce
# ╟─53984f21-33d9-46b8-b08c-d15e044f2752
# ╟─23608a2c-77cb-4401-b598-9a7eb3d8d469
# ╠═3df2b0f4-a7ae-4ad6-aad5-255330edf54b
# ╟─87cac64d-90a8-431c-8179-7f7462e5c11a
# ╠═ba1205e6-dcde-4c44-a246-5bc180085ec7
# ╟─89c2bc30-220c-4add-8bc8-21b5b610d45b
# ╟─bae4a3e4-2224-4e5d-ab15-bbdf86310ba8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
