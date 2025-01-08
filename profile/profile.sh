python -m cProfile -o myLog.profile sampler_phen-noise.py
gprof2dot -f pstats myLog.profile -o callingGraph.dot
dot -Tsvg callingGraph.dot > output.svg
