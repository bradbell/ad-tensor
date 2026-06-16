#! /usr/bin/env python3
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
import torch
import pathlib
#
# torch_path
torch_file = torch.__file__
torch_dir  = torch_file.replace('/__init__.py', '')
torch_path = pathlib.Path( torch_dir )
#
# found
found = None
for name in [ 'TorchConfig.cmake', 'torch-config.cmake' ] :
    for file in torch_path.rglob(name) :
        if found != None :
            print( found )
            print( file.absolute() )
            sys.exit( 'found multiple Torch cmake files; see output above' )
        found = file
#
# torch_dir
torch_dir = found.parent
print( torch_dir )
