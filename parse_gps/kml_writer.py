'''
Author: Cameron Ruddy 
Data: 12/01/2024
gmail: cruddy@research.Stevens.edu
'''

import csv

class kml_writer:

    def __init__(self, inpath, outpath):
        self.time_table = []
        self.coord_table = []


        with open(inpath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.time_table.append(row["Measurement_DateTime"])
                self.coord_table.append([row["GPS_lon"], row["GPS_lat"]])

        self.outfilehandle = open(outpath, 'w')
        self.outfilehandle.write(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        )
        self.outfilehandle.write(
            "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n"
        )
        
    def placemark(self, lat, lon, name, description):
        self.outfilehandle.write(
            "\t<Placemark>\n"
        )
        self.outfilehandle.write(
            "\t\t<name>{}</name>\n".format(
                name
            )
        )
        self.outfilehandle.write(
            "\t\t<description>{}</description>\n".format(
                description
            )
        )
        self.outfilehandle.write(
            "\t\t<Point>\n"
        )
        self.outfilehandle.write(
            "\t\t\t<coordinates>{},{},0</coordinates>\n".format(
                lon,lat
            )
        )
        self.outfilehandle.write(
            "\t\t</Point>\n"
        )
        self.outfilehandle.write(
            "\t</Placemark>\n"
        )

    def multiple_points(self, coord_list, name, description):
        
        self.outfilehandle.write(
            "\t<Placemark>\n"
        )
        self.outfilehandle.write(
            "\t\t<name>{}</name>\n".format(
                name
            )
        )
        self.outfilehandle.write(
            "\t\t<description>{}</description>\n".format(
                description
            )
        )
        
        for lon, lat in coord_list:
            self.outfilehandle.write(
                "\t\t<Point>\n"
            )
            self.outfilehandle.write(
                "\t\t\t<coordinates>{},{},0</coordinates>\n".format(
                    lon, lat
                )
            )
            self.outfilehandle.write(
                "\t\t</Point>\n"
            )
        self.outfilehandle.write(
            "\t</Placemark>\n"
        )

    def make_path(self, coord_list, **kwargs):

        ##### Header #####

        params = {
            "document_name":"",
            "document_desc":"",
            "line_color":"ff00ffff",
            "line_width":"5",
            "poly_color":"ff00ff00",
            "path_name":"",
            "path_desc":"",
            "extrude":"0",
            "tesselate":"1",
            "alt_mode":"clampToGround"
        }

        for key in kwargs.keys():
            if key in params.keys():
                params[key] = kwargs[key]

        header = ("\t<Document>\n"
            "\t\t<name>{}</name>\n"
            "\t\t<description>{}</description>\n"
            "\t\t<Style id=\"path_style\">\n"
            "\t\t\t<LineStyle>\n"
            "\t\t\t\t<color>{}</color>\n"
            "\t\t\t\t<width>{}</width>\n"
            "\t\t\t</LineStyle>\n"
            "\t\t\t<PolyStyle>\n"
            "\t\t\t\t<color>{}</color>\n"
            "\t\t\t</PolyStyle>\n"
            "\t\t</Style>\n"
            "\t\t<Placemark>\n"
            "\t\t\t<name>{}</name>\n"
            "\t\t\t<description>{}</description>\n"
            "\t\t\t<styleUrl>#path_style</styleUrl>\n"
            "\t\t\t<LineString>\n"
            "\t\t\t\t<extrude>{}</extrude>\n"
            "\t\t\t\t<tesselate>{}</tesselate>\n"
            "\t\t\t\t<altitudeMode>{}</altitudeMode>\n"
            "\t\t\t\t<coordinates>\n".format(
                params["document_name"],
                params["document_desc"],
                params["line_color"],
                params["line_width"],
                params["poly_color"],
                params["path_name"],
                params["path_desc"],
                params["extrude"],
                params["tesselate"],
                params["alt_mode"]

            ))

        self.outfilehandle.write(header)

        ##### Write Coords #####
        for lon, lat in coord_list:
            try:
                if float(lat) == 0.0 and float(lon) == 0.0:
                    continue
            except ValueError:
                pass

            try:
                self.outfilehandle.write(
                    "\t\t\t\t\t{},{},0\n".format(
                        lon,lat
                    )
                )
            except IndexError:
                break
            
        ##### Write Footer #####
        footer = ("\t\t\t\t</coordinates>\n"
                  "\t\t\t</LineString>\n"
                  "\t\t</Placemark>\n"
                  "\t</Document>\n"
                  "</kml>")


        self.outfilehandle.write(footer)
