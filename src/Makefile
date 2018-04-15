BUILDDIR=../build

all: mk-som mk-bsom

build:
	mkdir -p $(BUILDDIR)

mk-som: build
	make -C som/ all

mk-bsom: build
	make -C bsom/ all

debug: build
	make -C som/ debug
	make -C bsom/ debug

clean:
	make -C som/ clean
	make -C bsom/ clean
