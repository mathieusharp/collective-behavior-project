# Collective Behavior Project: from predator to shepherd

## Team (Group F)

| Name | GitHub |
|------|--------|
| Mathieu Bonnet | [@mathieusharp](https://github.com/mathieusharp) |
| Loup Benois | [@loulliam](https://github.com/loulliam) |
| Alexandre Peret | [@Elysios05](https://github.com/Elysios05) |

## Starting point

Klamser & Romanczuk (2021). Collective predator evasion: Putting the criticality hypothesis to the test. PLoS Computational Biology. [doi:10.1371/journal.pcbi.1008832](https://doi.org/10.1371/journal.pcbi.1008832), [original code](https://github.com/PaPeK/PredatorPrey).

The paper simulates a 2D school of fish attacked by a predator. The predator catches the fewest fish when the school is halfway between disordered and fully aligned.

## Project description

We reimplement the model in Python and reproduce its main result. We then improve the predator with one behavioural change, such as its attack angle, and also test whether the result holds in 3D. Finally, we turn the predator into a shepherd that steers the school without breaking it apart, trying to adapt to fish the model of [Strömbom et al. (2014)](https://doi.org/10.1098/rsif.2014.0719) (a single sheepdog herding a flock of sheep).

## Plan / milestones

- Report 1 (13 Nov 2026): literature review, Python reimplementation, reproduction of the main result.
- Report 2 (11 Dec 2026): improved predator, 3D version of the model, comparison with the literature.
- Report 3 (8 Jan 2027): shepherd, compared with [Van Havermaet et al. (2024)](https://doi.org/10.1038/s41598-024-65894-5) (a group of robots that encircle a fish school to guide it); final report.
