#! /usr/bin/env python
import sys
import os
import sferes
sys.path.insert(0, sys.path[0]+'/waf_tools')
print sys.path[0]


from waflib.Configure import conf



def build(bld):
	
	# simple_ebn
	bld.program(features = 'cxx',source = 'ex_simple.cpp',includes = '. ../../',uselib = 'TBB BOOST EIGEN PTHREAD MPI',use = 'sferes2',target = 'ex_simple')


	# simple_ebn
	bld.program(features = 'cxx',source = 'ex_simple_div.cpp',includes = '. ../../',uselib = 'TBB BOOST EIGEN PTHREAD MPI',use = 'sferes2',target = 'ex_simple_div')

	# bld.program(features = 'cxx',source = 'ex_simple_div_tanh.cpp',includes = '. ../../',uselib = 'TBB BOOST EIGEN PTHREAD MPI',use = 'sferes2',target = 'ex_simple_div_tanh')


	# bld.program(features = 'cxx',source = 'ex_simple_div_sb.cpp',includes = '. ../../',uselib = 'TBB BOOST EIGEN PTHREAD MPI',use = 'sferes2',target = 'ex_simple_div_sb')
