USER = jessecusack

MITGCM_DIR = /Users/Cusack/MITgcm
OPTFILE = /Users/Cusack/MITgcm/tools/build_options/SIOiMac

.PHONY: all link build

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all           to run all processing steps"
	@echo "  link          to generate directories and link files"
	@echo "  build         to build the model"
	@echo "  clean         to clean up everything inc. directories run and build"


# Group some processing steps
all: link build
link: make_dirs link_input
build: make_dirs genmake buildmodel

###########################
# LINK INPUT FILES TO RUN #
###########################

# set up directories
make_dirs:
	@echo "generating directories on scratch drive"
	@mkdir -p build/
	@mkdir -p run/

# link input files and utils
DATA_FILES = $(abspath $(wildcard input/data*))
MORE_FILES = $(abspath $(wildcard input/OB*) input/eedata input/Tref input/T.init $(wildcard input/del*) input/topogSamoa.bin $(wildcard input/OB_*) $(wildcard input/V*.bin))
CODE_FILES = $(abspath $(wildcard code/*))

link_input:
	@echo "copying input files to run directory"
	@cp $(DATA_FILES) run/
	@cp $(MORE_FILES) run/
	@echo "linking code files to build directory"
	@ln -fs $(CODE_FILES) build/

# generate the makefile
genmake:
	@echo "running genmake"
	@$(MITGCM_DIR)/tools/./genmake2 -rd=$(MITGCM_DIR) -of=$(OPTFILE) -mods=$(CURDIR)/code/ -mf=$(CURDIR)/build/Makefile -mpi
	@echo "moving files"
	@mv AD_CONFIG.h PACKAGES_CONFIG.h genmake.log build/

# build the model
buildmodel:
	@echo "building model"
	@cd build/ && make -j 4 depend && make -j 4
	@echo "link executable to run directory"
	@cp build/mitgcmuv run/

# clean stuff up
clean:
	@echo "cleaning everything (meaning EVERYTHING!)"
	@cd build/ && make CLEAN && cd ../
	@rm -rf run/ build/
