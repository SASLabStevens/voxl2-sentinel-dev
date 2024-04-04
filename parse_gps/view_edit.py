'''
Author: Cameron Ruddy 
Data: 12/01/2024
gmail: cruddy@research.Stevens.edu
'''

from colorama import Fore, Back, Style
import pprint, os, sys, json

USER_MOD_COLOR = Fore.CYAN
RESET = Fore.RESET

class view_editor:
    
    def __init__(self):
        self.params = {
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

        self.user_params = {
            "document_name":"",
            "document_desc":"",
            "line_color":"red",
            "line_opacity":"100",
            "line_width":"5",
            "poly_color":"green",
            "poly_opacity":"100",
            "path_name":"",
            "path_desc":"",
            "extrude":"0",
            "tesselate":"1",
            "alt_mode":"clampToGround"
        }

        # For some reason KML reverses the color codes
        self.valid_colors = {
            "red" : "0000FF",
            "blue" : "FF0000",
            "green" : "00FF00",
            "yellow" : "00FFFF",
            "purple": "800080",
            "orange": "0080FF",
            "pink": "FF00FF"
        }
    
    def get_user_input(self):
        user_input = ""
        while(user_input != "finish"):
            os.system("clear")
            print("Current arguments:")
            pprint.pprint(self.user_params)
            user_input = input("\nSpecify field to edit, \"finish\", or \"exit\": ")
            if user_input == "exit":
                sys.exit(0)
            elif user_input in self.user_params.keys():

                if user_input == "line_color" or user_input == "poly_color":
                    print("Valid colors are: ")
                    for key in self.valid_colors.keys():
                        print(key)

                    new_value = input("Modify {}\"{}\"{} from {}\"{}\"{} to: {}".format(
                        USER_MOD_COLOR, user_input, RESET,
                        USER_MOD_COLOR, self.user_params[user_input], RESET,
                        USER_MOD_COLOR
                    ))

                    if new_value not in self.valid_colors.keys():
                        print(RESET)
                        continue

                else:
                    new_value = input("Modify {}\"{}\"{} from {}\"{}\"{} to: {}".format(
                        USER_MOD_COLOR, user_input, RESET,
                        USER_MOD_COLOR, self.user_params[user_input], RESET,
                        USER_MOD_COLOR
                    ))
                    
                print(RESET)
                self.user_params[user_input] = new_value

        self.format_output()

    def format_color(self, color):
        try:
            return self.valid_colors[color]
        except KeyError:
            print("{}Warning{}: {} is not a valid color, using default".format(
                Fore.YELLOW, RESET, color
            ))
            return self.valid_colors["red"]
        
    def format_opacity(self, opacity):
        scaled_vec = list(range(100))
        step = 256/100
        for n in scaled_vec:
            val = int(n*step)
            scaled_vec[n] = "%0.2X" % val
        scaled_vec[99] = "FF"

        return scaled_vec[int(opacity)-1]
    
    def format_output(self):
        for key in self.params.keys():
            if key == "line_color":
                self.params[key] = "{}{}".format(
                    self.format_opacity(self.user_params["line_opacity"]),
                    self.format_color(self.user_params["line_color"])
                )
            elif key == "poly_color":
                self.params[key] = "{}{}".format(
                    self.format_opacity(self.user_params["poly_opacity"]),
                    self.format_color(self.user_params["poly_color"])
                )
            else:
                self.params[key] = self.user_params[key]



    
