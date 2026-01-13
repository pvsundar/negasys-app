# SmartTVs 4'Us: Dyadic Household Product-Line Design

**A Teaching Case for Product-Line Optimization with NEGASYS**

Version 5.0 \| January 2026

## P.V. (Sundar) Balakrishnan

#### School of Business, University of Washington Bothell 

---

## Case Summary

This case places students in the role of an analytics team advising Dana
Briggs, Senior Category Manager at MegaMart, on smart TV assortment for a
new retail store. Students use the NEGASYS decision-support tool to
optimize product lines while modeling how dual-partner households
negotiate purchases using four alternative decision rules: linear weighted
utility, symmetric Nash bargaining, generalized Nash with bargaining
power, and Rawlsian minimum.

**Key Learning Objectives:** - Interpret dyadic conjoint data with two
utility vectors per household - Compare household decision rules and their
behavioral implications - Use optimization software under different
modeling assumptions - Evaluate robustness of recommendations across
decision frameworks

--------------------------------------------------------------------------

## 1. The Meeting at MegaMart

On a gray February morning, **Dana Briggs**, Senior Category Manager for
Consumer Electronics at **MegaMart**, stared at the assortment planning
spreadsheet on her screen. In six weeks, MegaMart would open a new
suburban "showcase" store that was expected to anchor its presence in a
fast-growing market. The executive team wanted the store to signal
MegaMart's strength in higher-margin, higher-involvement categories such
as smart televisions.

The constraint was simple to explain, yet hard to manage: the **SmartTVs
4'Us wall could only hold a handful of distinct models**. Competing
retailers, both brick-and-mortar and online, already carried dozens of
combinations of brands, sizes, and feature sets. Dana had to recommend a
**small, tightly curated product line** that would:

-   Appeal to couples and families making a joint purchase;
-   Grow MegaMart's market share in the local catchment area;
-   Protect or improve category margins;
-   Be operationally realistic for store staff and inventory planners.

As she scanned the list of candidate TVs, Dana saw competing priorities
everywhere. Larger screens and premium panels were attractive but risky at
higher price points; compact models drove traffic but delivered thin
margins; gaming features mattered a great deal to some households and not
at all to others. The question was no longer "Which TV models do customers
like?" but **"Which compact set of models will real households,
negotiating together, actually choose?"**

MegaMart had recently invested in a substantial market research study. The
company now had **individual-level conjoint utilities for both members of
each household** in the target market. Dana suspected that these data
could provide a more disciplined way to design the assortment,
particularly if they could be used to model how couples reconcile their
differences when choosing a TV together.

Dana called a 10 a.m. meeting.

### 1.1 Calling in the Analytics and Marketing Team

At 10:00 sharp, the small conference room filled up. Dana opened the
meeting.

> "Our SmartTVs 4'Us wall is prime real estate. I need a recommendation on
> which models to carry and how many. We have limited space, a competitive
> local market, and couples who do not always agree on what they want. I
> would like us to use the conjoint study rather than gut feel. By next
> week, I want a clear, data-based product-line recommendation under
> realistic assumptions about how households decide."

Seated around the table were:

-   **Alex Chen**, Senior Data Analyst, responsible for choice modeling
    and decision-support tools;
-   **Maria Lopez**, Marketing Insights Manager, who had overseen the
    conjoint research;
-   **Jayanti Arulmugam**, Promotions and Merchandising Lead for the new
    store;
-   **Malik Okafor**, a junior analyst who had been experimenting with a
    new in-house decision-support system, nicknamed **NEGASYS**, for
    negotiated household product-line design.

Dana turned to Maria.

> "Before we talk about product lines, can you give us the thirty-second
> version of what this conjoint study actually produced?"

Maria nodded.

> "We recruited couples in our target market who said they expected to buy
> a new primary living-room TV in the next year. Each partner completed
> the same survey separately. They rated and chose between hypothetical
> smart TV profiles that varied in brand family, screen size and display
> technology, audio configuration, parental controls, smart platform, and
> price band.
>
> From those responses, our vendor estimated **part-worth utilities** for
> every attribute level for each individual. Higher part-worths mean
> stronger preference. For each person, the utilities across all attribute
> levels are normalized to a common scale.
>
> The key point is that for every couple, we now have **two utility
> vectors**, one for each partner. The technical details, estimation
> procedure, and examples of the part-worths are summarized in **Appendix
> A**. The data layout and coding are described in **Appendix B**."

Dana looked over to Alex.

> "So far, we have always treated these utilities as if we had a single
> decision maker. For this store, that is not realistic. I want to
> understand how different **household decision rules** might change the
> product line we recommend. Also, I keep hearing about this NEGASYS tool.
> What exactly does it do for us?"

### 1.2 What is a Product-Line Optimizer?

Alex leaned forward.

> "NEGASYS is essentially a **product-line optimizer**. Think of it as a
> decision-support engine that reads in our conjoint utilities and then
> searches through thousands of possible assortments to identify a **small
> set of products** that performs best against a specified objective.
>
> We tell it: here is the design space defined by our attributes and
> levels; here is the capacity constraint on how many models we can carry;
> here is how households evaluate each product; and here is what we want
> to optimize, such as expected profit or market share.
>
> The optimizer then evaluates candidate product lines, predicts how
> households would choose among those lines and their current TVs,
> aggregates the choices into market shares and profits, and reports back
> the best lines it finds under those assumptions."

Malik added:

> "The important twist for this case is that NEGASYS does this **at the
> household level**. It treats alternating rows in the utility data as
> partners in the same household and uses a decision rule to aggregate
> their two utility vectors into a single joint evaluation for each TV.
> The optimization itself looks like a conventional product-line search;
> what is new here is how we model the **joint choice of the couple**."

Dana nodded.

> "So the optimizer is the engine; the real question is how we define the
> **household utility function** that feeds that engine. Let us talk about
> that."

### 1.3 Brainstorming the Household Decision Rules

Maria spoke first.

> "Historically, when we had any kind of 'two person' situation, we just
> used a **linear model**: we took a convex combination of the two
> utilities, something like
>
> joint utility = α × utility of Partner H + (1 − α) × utility of Partner
> W.
>
> If we did not know who had more influence, we set α to one half. It is
> simple to explain to managers; it treats the joint evaluation as an
> average with adjustable weights."

Alex nodded.

> "That linear weighted rule is still our **default** in NEGASYS. It is
> intuitive: if one partner has more say in electronics purchases, we can
> give them a larger weight. But given that we now have dyadic data and we
> care about fairness and negotiation, we could be more ambitious."

Malik looked up from his notebook.

> "I have been reading about the **Nash bargaining solution**. One way to
> model the household is to look at the gain each partner gets from
> switching from their status quo TV to a new TV and then take the
> **product of those gains**. That tends to favor options where **both
> partners gain**, and it penalizes options that make one partner much
> better off and the other worse off. NEGASYS can already implement that
> Nash product rule."

Dana raised an eyebrow.

> "So we have at least two candidates: a convex combination and a
> symmetric Nash product. Does that fully reflect how couples actually
> bargain?"

Jayanti shook her head.

> "Not always. In practice, one partner often has **more power**: maybe
> they control the budget; maybe they care more about tech; maybe they are
> the one who will watch more sports. Could we build in **relative power**
> somehow?"

Alex smiled.

> "That is exactly what a **generalized Nash bargaining model** does.
> Instead of taking a simple product of gains, we raise each partner's
> gain to a power that reflects their bargaining weight: a parameter α for
> Partner H and (1 − α) for Partner W.
>
> If α is greater than one half, Partner H has more influence; if α is
> less than one half, Partner W does. It is still a cooperative bargaining
> model; it just allows **asymmetric power**."

There was a pause, then Jayanti laughed.

> "This reminds me of my MBA night class. We just read John Rawls's *A
> Theory of Justice*. What if we took a **Rawlsian view** inside the
> household: we pick the TV that maximizes the utility of the **least
> satisfied partner**. In other words, we care most about not leaving one
> partner behind; the household prefers options that are fairest in that
> sense."

Malik nodded enthusiastically.

> "We can code that as a **Rawlsian minimum rule**: for each TV profile,
> we look at both partners' utilities and treat the household's evaluation
> as the **minimum** of the two. The optimizer then searches for product
> lines that perform well when each couple ranks TVs based on that
> least-satisfied-member utility."

Dana looked around the table.

> "So, in summary, we have at least **four** plausible ways to model
> household decisions:
>
> 1.  A **linear weighted utility**: our historical default;\
> 2.  A **symmetric Nash bargaining solution**, using the product of gains
>     relative to the status quo;\
> 3.  A **generalized Nash bargaining model** with adjustable bargaining
>     power;\
> 4.  A **Rawlsian minimum rule** that protects the worst-off partner.
>
> For each of these, NEGASYS can search for the best product line under
> that rule and report back the results. Our job is to see how sensitive
> the recommended assortment is to the choice of rule, and then make a
> recommendation we can defend."

The team agreed to reconvene later in the week with preliminary results.

--------------------------------------------------------------------------

## 2. The Conjoint Study

### 2.1 Sample and Recruitment

MegaMart's market research vendor recruited **200 individuals**, forming
**100 dual-partner households**, from the store's target trading area.
Eligibility criteria included:

-   Both partners in the household agreed to participate;
-   The household reported being "very likely" or "somewhat likely" to
    purchase a new primary living-room TV within the next 12 months;
-   The household was the primary decision-making unit for major
    electronics purchases.

Each partner completed the survey **independently**, in randomized order,
with instructions not to discuss specific questions with their spouse or
partner until after completion.

### 2.2 Conjoint Tasks

Respondents completed a series of discrete-choice tasks:

-   Each task presented **3-4 smart TV profiles** plus a "none / keep my
    current TV" option;
-   Profiles varied in **brand family, screen size and display technology,
    audio configuration, parental controls, smart platform, and price
    band**;
-   The design followed standard conjoint principles, using a fractional
    factorial or efficient design so that attribute levels were rotated
    and balanced across tasks.

Each individual completed a modest number of tasks, sufficient to support
individual-level estimation while keeping respondent burden manageable.

### 2.3 Individual-Level Utilities and Dyadic Structure

The research vendor estimated **individual-level part-worth utilities**
for each attribute level using standard techniques such as hierarchical
Bayesian multinomial logit. For each individual:

-   A vector of part-worth utilities is estimated, one value per attribute
    level;
-   Higher values indicate stronger preference; differences across levels
    within an attribute are meaningful;
-   Utilities are scaled to a common range for interpretability.

The resulting dyadic dataset has the following structure:

-   One line of utilities per individual;
-   **Alternating lines correspond to members of the same household**
    -   Line 1: Household 1, Partner H (for example, Husband or Partner A)
    -   Line 2: Household 1, Partner W (Wife or Partner B)
    -   Line 3: Household 2, Partner H
    -   Line 4: Household 2, Partner W; and so on.

Students can think of the two lines for a household as a pair of utility
vectors that describe, for that couple, how each partner values the
different smart TV features.

The technical details of the conjoint design and estimation, along with an
illustrative slice of the utility data, are provided in **Appendix A** and
**Appendix B**.

--------------------------------------------------------------------------

## 3. Smart TV Attributes and Levels

Based on secondary research, internal sales data, and a set of pretests,
the team narrowed the relevant feature space for this market to the
following **six attributes**.

These attributes define the **design space** that NEGASYS uses when
generating and evaluating candidate smart TV models.

### 3.1 Attribute Definitions

| Attribute | Levels |
|-------------------------------------------|-------------------------------|
| **Brand family** (3 levels) | ValueMax (value-oriented private brand), MainStreet (mainstream national brand), PrestigeView (premium global brand) |
| **Screen size and display** (3 levels) | 50" 4K LED, 55" 4K QLED, 65" 4K QLED thin bezel |
| **Audio configuration** (3 levels) | Integrated TV speakers, TV + 2.1 soundbar bundle, TV + immersive 5.1 sound system |
| **Parental controls** (2 levels) | Basic profiles only, Advanced kids mode & parental controls |
| **Smart platform** (2 levels) | Core streaming apps, Streaming + voice assistant & personalized guide |
| **Price band** (4 levels) | Entry, Value, Premium, Ultra-premium |

A product profile is defined by one level from each attribute. For
example:

-   ValueMax, 55" 4K QLED, TV + 2.1 soundbar bundle, basic profiles only,
    core streaming apps, value price band; or\
-   PrestigeView, 65" 4K QLED thin bezel, TV + immersive 5.1 system,
    advanced kids mode & parental controls, streaming + voice assistant &
    personalized guide, premium price band.

The attributes and levels are summarized for students in **Appendix B**.

--------------------------------------------------------------------------

## 4. Competitive Offerings and the Status Quo

The dyadic conjoint study included questions about each household's
**current living-room TV**. In addition, MegaMart's insights team mapped
the most common TVs in the local market into the same attribute structure.

### 4.1 Competitor Set

The competitor set is the collection of smart TV models that are already
available at other retailers in the local market. For the purposes of
NEGASYS, this set:

-   Is encoded using the same six attributes and levels as the conjoint
    design;
-   Represents the "outside options" that households can continue to buy
    elsewhere if MegaMart's assortment is not attractive;
-   Is used to define each household's **status quo** product.

Students will receive the competitor set in the form of a `.01c` file,
which lists each competitor TV's attribute levels in the same format as
the utility file. The competitor file is documented in **Appendix B**.

### 4.2 Status Quo as Disagreement Point

In household bargaining models, the **status quo** or **disagreement
point** is the outcome that prevails if the household does not reach
agreement on a new purchase. For this case:

-   The status quo is the household's **current living-room TV**, mapped
    to one of the competitor products.
-   If no product in MegaMart's line offers sufficient joint utility, the
    household keeps its current TV.

For the case, students can treat the status quo as given; NEGASYS reads
the status quo information and includes it in all household choice
calculations.

--------------------------------------------------------------------------

## 5. Household Decision Rules

When NEGASYS evaluates a candidate TV profile for a household, it must
translate the utilities of the two partners into a single **joint
evaluation**. The four rules discussed in Dana's meeting correspond to
four different ways of thinking about fairness, power, and compromise
inside the household.

### 5.1 Linear Weighted Utility

The **linear weighted rule** treats household utility as a convex
combination of the two partners' utilities. A parameter α represents
Partner H's influence; (1 − α) represents Partner W's influence.

This has been MegaMart's default rule:

-   If α equals one half: both partners are equally influential.
-   If α is greater than one half: Partner H has more say.
-   The rule is easy to explain and calibrate; it matches managers'
    intuition that some people simply "have more say" in certain
    categories.

### 5.2 Symmetric Nash Bargaining Solution

The **Nash bargaining rule** focuses on each partner's gain from switching
relative to the status quo TV.

-   For a given product, each partner's utility gain is computed relative
    to their utility for the current TV.
-   If both partners gain, the joint evaluation is proportional to the
    **product** of these gains.
-   If one partner loses, the product becomes small or negative; such
    profiles are heavily penalized.

Intuitively, this rule rewards products that make both partners better off
and punishes those that sacrifice one partner to benefit the other.

### 5.3 Generalized Nash with Bargaining Power

The **generalized Nash rule** extends the symmetric Nash solution by
allowing **unequal bargaining power**.

-   Each partner's gain is raised to an exponent that reflects their
    bargaining weight: a parameter α for Partner H and (1 − α) for Partner
    W.  
-   When α is greater than one half, Partner H's gain counts more in the
    joint evaluation; when α is less than one half, Partner W's gain
    counts more.

This rule captures situations where, for example, one partner controls the
budget or has stronger preferences in the electronics category.

### 5.4 Rawlsian Minimum Rule

The **Rawlsian rule** evaluates each product by looking at the **least
satisfied partner**.

-   For a candidate TV, NEGASYS looks at both partners' utilities and
    takes the lower of the two as the household's evaluation.
-   A product that leaves one partner far behind will receive a low joint
    score even if the other partner is delighted.

This rule reflects a fairness-first philosophy: the household prioritizes
the welfare of the worst-off partner.

### 5.5 Choice Behavior and Comparison

Under any of these rules, the process for a household is:

1.  Compute the joint evaluation of each candidate TV and of the status
    quo;
2.  Choose the option with the highest joint evaluation;
3.  Switch from the status quo only if at least one new TV beats it on
    joint utility.

By running NEGASYS under different rules, Dana's team can assess:

-   How sensitive the recommended assortment is to assumptions about
    intra-household bargaining;
-   Whether certain products emerge as robust across rules;
-   How trade-offs among profit, market share, and fairness shift when the
    household rule changes.

Formal definitions and simple numerical illustrations for each rule appear
in **Appendix C**.

--------------------------------------------------------------------------

## 6. Product-Line Design Problem for the New Store

Dana's decision can be summarized as a **product-line design problem**
with:

-   A **design space** defined by the attributes and levels in Section 3;
-   A **demand model** that predicts household choices from dyadic
    utilities using one of the four rules in Section 5;
-   A **status quo option** for each household drawn from the competitor
    set;
-   A **capacity constraint** on the number of smart TV models the new
    store can display;
-   A profit model that combines predicted unit sales with per-unit
    margins for each profile.

### 6.1 Decision Variables

Dana must decide:

1.  **How many distinct smart TV models** to carry in the new store;
2.  **Which specific combinations of attribute levels** these models
    should have.

Within NEGASYS, these decisions are represented as a product-line
configuration: a set of product profiles selected from the design space,
subject to a maximum line size.

### 6.2 Objectives and Trade-offs

The executive team's objectives are multifaceted:

-   Increase **category market share** in the local catchment area;
-   Achieve an attractive **profit contribution** from the smart TV
    category;
-   Ensure that the assortment feels coherent and understandable to
    customers and to store staff;
-   Avoid assortments that systematically favor one member of the couple
    at the expense of the other.

NEGASYS can be run with different objective functions; for example:

-   Maximize expected profit;
-   Maximize expected market share subject to a profit floor;
-   Maximize a hybrid objective that combines profit with a fairness
    metric.

For this case, students are asked to focus on a profit-based objective
with a fixed line size, then explore how robust the recommended line is
across decision rules.

--------------------------------------------------------------------------

## 7. The Student's Task

Dana closes the meeting with a clear directive to the team:

> "Using the dyadic conjoint data, the competitor set, and the NEGASYS
> product-line optimizer, I want you to recommend a smart TV product line
> for our new store. I want to see the analysis under each of the four
> household decision rules we discussed. Your job is not only to propose
> an assortment, but also to tell me how much that recommendation depends
> on how we think couples actually negotiate. When we stand in front of
> the executive team, I want us to be able to say: 'Here is the line we
> propose, here is how well it performs, and here is how it holds up under
> different views of household decision making.'"

Students are placed in the role of Dana's analytics and marketing team.
Using the case narrative, the appendices, and the accompanying data and
software tools, they are asked to:

-   Interpret the dyadic conjoint utilities;
-   Understand the four household decision rules;
-   Run NEGASYS or an equivalent product-line optimization procedure under
    different rules;
-   Recommend a product line and justify it in terms of performance,
    fairness, and robustness.

A suggested assignment structure, deliverables, and optional extensions
are provided separately in the Teaching Note and may be adapted to the
course level and available software.

--------------------------------------------------------------------------

## Appendices (see separate files)

-   **Appendix A**: Conjoint Study Design and Utility File Documentation
-   **Appendix B**: Competitor Set and Status Quo Documentation\
-   **Appendix C**: Household Decision Rules with Formal Definitions and
    Worked Example

--------------------------------------------------------------------------

*Case prepared for educational use. P. V. Sundar Balakrishnan and
collaborators.*
