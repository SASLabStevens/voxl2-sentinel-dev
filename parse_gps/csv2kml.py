'''
Author: Cameron Ruddy 
Data: 12/01/2024
gmail: cruddy@research.Stevens.edu
'''

import csv, os
import kml_writer, view_edit
import argparse
import pdb

parser = argparse.ArgumentParser(
    prog="CSV to KML",
    description="Turns CSV coordinates into a KML file"
)

parser.add_argument('-i',
                    '--infile',
                    help="Path to CSV file",
                    metavar="",
                    action="store")

parser.add_argument('-o',
                    '--outfile',
                    help="Write output kml to this path",
                    metavar="",
                    action="store")

args = parser.parse_args()

if args.outfile == None:
    outfile = os.path.basename(args.infile)[:-4] + ".kml"

else:
    text,ext = os.path.splitext(args.outfile)
    if ext == ".kml":
        outfile = args.outfile
    else:
        outfile = text + ".kml"
    

session = kml_writer.kml_writer(args.infile, 
                outfile)

editor = view_edit.view_editor()
editor.get_user_input()

session.make_path(session.coord_table,**(editor.params))
