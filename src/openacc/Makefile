all: heat-2d

heat-2d: heat-2d.c
	pgcc -acc -O3 -DTIME -DVERIFY -Minfo=accel -ta=nvidia,cc35 $< ~/forge/pgiwrapper/libpgiwrapper.a -o $@

clean:
	rm -rf heat-2d

