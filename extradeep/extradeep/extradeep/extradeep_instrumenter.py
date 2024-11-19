# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
import logging
import os
import sys
import shutil
import fnmatch

import extradeep

def add_nvtx_import_statement(code):
    nvtx_package_import = "import nvtx\n"
    code.insert(0, nvtx_package_import)
    return code

def add_nvtx_decorators(code, function_definitions, parent):
    for i in range(len(function_definitions)):
        func = function_definitions[i]
        call_stack_func_name = func[0]
        line_number = func[1] - 1 # -1 because of list indexing 
        line_string = code[line_number]
        #print(line_number)
        # when there is a tf function declaration move the decorator to the line before that
        if "@tf.function" in code[line_number - 1]:
            line_number -= 1
        #print(line_number)
        if parent == "":
            decorator_string = "@nvtx.annotate(message=\""+call_stack_func_name+"\")\n"
        else:
            decorator_string = "@nvtx.annotate(message=\""+parent+"->"+call_stack_func_name+"\")\n"
        new_line = add_leading_spaces(line_string, decorator_string)
        #print(line_string)
        #print(new_line)
        code.insert(line_number, new_line)
        # update the code line indexes for the remaining functions declarations
        counter_index = i+1
        while counter_index < len(function_definitions):
            start_line = function_definitions[counter_index][1]
            start_line += 1
            end_line = function_definitions[counter_index][2]
            end_line += 1
            function_definitions[counter_index] = (function_definitions[counter_index][0], start_line, end_line)
            counter_index += 1
    return code

def write_code(new_path, code):
    f = open(new_path, "w")
    for x in code:
        f.write(x)
    f.close()

def get_parent_of_object_declaration(module_line_numbers, function_definitions):
    for i in range(len(module_line_numbers)):
        #print(module_line_numbers[i])
        line_number = module_line_numbers[i][1][0]
        #print(line_number)
        function_name = ""
        for j in range(len(function_definitions)):
            start = function_definitions[j][1]
            end = function_definitions[j][2]
            if line_number > start and line_number <= end:
                function_name = function_definitions[j][0]
                break
        module_line_numbers[i] = (module_line_numbers[i][0][0], module_line_numbers[i][0][1], module_line_numbers[i][1][0], function_name)
    return module_line_numbers

def read_code(path):
    f = open(path, "r")
    code = []
    for x in f:
        code.append(x)
    f.close()
    return code

def rec_func(list, current_item):
    in_list = False
    new_item = None
    for j in range(len(list)):
        if current_item[0] == list[j][0]:
            pass
        else:
            #print(current_item)
            if current_item[1] > list[j][1] and current_item[2] <= list[j][2]:
                #print(current_item[0]+" is inside of "+list[j][0])
                new_item = list[j]
                in_list = True
    if in_list == True:
        return rec_func(list, new_item) + "->" + current_item[0]
    else:
        return current_item[0]

def recursive_check_function_in_function_declarations(function_definitions):
    names = []
    for i in range(len(function_definitions)):
        function_definition = function_definitions[i]
        name = rec_func(function_definitions, function_definition)
        #print("rec func return:",name)
        names.append(name)
    for i in range(len(names)):
        function_definitions[i] = (names[i], function_definitions[i][1], function_definitions[i][2])
    return function_definitions

def get_function_definitions(code):
    function_definitions = []
    for i in range(len(code)):
        if "def " in code[i]:
            #print(code[i])
            function_name = code[i]
            function_name = function_name.replace("def ", "")
            pos = function_name.find("(")
            function_name = function_name[:pos]
            function_name = function_name.replace(" ", "")
            number_leading_spaces = get_leading_spaces(code[i])
            function_start_line = i+1
            #print("start:",function_start_line)
            function_end_line = None
            counter = function_start_line+1
            while counter < len(code):
                line = code[counter]
                nr_leading_spaces = get_leading_spaces(line)
                if nr_leading_spaces <= number_leading_spaces:
                    if line == "\n" or line.isspace():
                        pass
                    else:
                        function_end_line = counter
                        break
                #print(code[len(code)-1])
                counter += 1
            #print(counter)
            if counter == len(code) and function_end_line is None:
                function_end_line = len(code) - 1
            #print("end:",function_end_line)
            # name, start, end
            function_definitions.append((function_name, function_start_line, function_end_line))
    return function_definitions

def get_line_numbers_class_instances(module_imports, code):
    module_line_numbers = []
    for x in module_imports:
        lines = []
        for j in range(len(code)):
            if x[0] in code[j]:
                if "from" in code[j] or "import" in code[j]:
                    pass
                else:
                    lines.append(j)
                    # INFO: on purpose only retrieve the first declaration for now
                    break
        module_line_numbers.append((x,lines))
    return module_line_numbers

def get_modules_and_paths(code):
    modules = []
    from_statements = []
    import_statements = []
    for i in range(len(code)):
        if fnmatch.fnmatch(code[i], "*from*import*") and "#" not in code[i]:
        #if "from " in code[i] and "#" not in code[i] and "\"\"\"" not in code[i-1]:
            import_statement = code[i]
            import_statements.append(import_statement)
            pos = import_statement.find("import ")
            from_statement = import_statement[:pos-1]
            from_statement = from_statement.replace("from ", "")
            #print(from_statement)
            from_statements.append(from_statement)
            import_statement = import_statement[pos+1+len("import"):]
            if "as" in import_statement:
                import_statement = import_statement.split(" ")
                import_statement = import_statement[len(import_statement)-1]
                import_statement = import_statement.replace("\n", "")
            else:
                #print("DEBUG:",import_statement)
                # INFO: on purpose just get the first declaration here for now...
                if "," in import_statement:
                    pos = import_statement.find(",")
                    import_statement = import_statement[:pos]
                #print("DEBUG:",import_statement)
                import_statement = import_statement.replace("\n", "")
                modules.append(import_statement)
    module_imports = []
    #print("DEBUG:",modules)
    #print("DEBUG:",from_statements)
    #print("DEBUG:",import_statements)
    for i in range(len(modules)):
        x = (modules[i], from_statements[i])
        module_imports.append(x)
    return module_imports

def get_leading_spaces(line_string):
    number_leading_spaces = len(line_string) - len(line_string.lstrip(" "))
    return number_leading_spaces

def add_leading_spaces(line_string, annotation_decorator):
    number_leading_spaces = len(line_string) - len(line_string.lstrip(" "))
    leading_spaces = ""
    for _ in range(number_leading_spaces):
        leading_spaces += " "
    annotation_decorator = leading_spaces + annotation_decorator
    return annotation_decorator

def create_folder_for_output(output_path):
    new_folder_path = ""
    # if default output path, (delete and) create new folder for instrumented code
    if output_path == "instrumented_code":
        current_path = os.getcwd()
        new_folder_path = os.path.join(current_path, output_path)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        else:
            shutil.rmtree(new_folder_path)
            os.makedirs(new_folder_path)
    # if user path, (delete and) create new folder for instrumented code
    else:
        if os.path.exists(output_path):
            new_folder_path = os.path.join(output_path, "instrumented_code")
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            else:
                shutil.rmtree(new_folder_path)
                os.makedirs(new_folder_path)
        else:
            logging.error("The specified output path does not exist.")
            sys.exit(1)
    return new_folder_path

def main(args=None, prog=None):
    """
    main function that runs the extradeep code instrumenter

    :param args: parameter for arguments entered in the terminal
    :param prog: parameter for programm
    """

    # Define argparse commands for input, output operations
    parser = argparse.ArgumentParser(prog=prog, description="Extra-Deep instrumenter, instrument your deep learning code automatically.\nAll functions will be instrumented automatically. You need to specify an output directory where the instrumented version of your code will be saved.", add_help=False)
    positional_arguments = parser.add_argument_group("Positional arguments")

    # Define basic program arguements such as log, help, and version outputs
    basic_arguments = parser.add_argument_group("Optional arguments")
    basic_arguments.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
    basic_arguments.add_argument("-v", "--version", action="version", version=extradeep.__title__ + " " + extradeep.__version__, help="Show program's version number and exit.")
    basic_arguments.add_argument("--log", action="store", dest="log_level", type=str.lower, default='warning', choices=['debug', 'info', 'warning', 'error', 'critical'], help="Set program's log level (default: warning).")

    # Define the path argument that points to the folder the data will be loaded from
    positional_arguments.add_argument("path", metavar="FILE_PATH", type=str, action="store",
                                      help="Specify the main file path for instrumenter to work with")
    positional_arguments.add_argument('--out', dest='out', action='store',type=str, default="instrumented_code",
                                      help='Set the output directory for the instrumented code. The insntrumenter will create a new folder in this directory called instrumented_code where you will find the result.')

    # parse args
    arguments = parser.parse_args(args)

    # set log level
    loglevel = logging.getLevelName(arguments.log_level.upper())

    # set log format location etc.
    if loglevel == logging.DEBUG:
        logging.basicConfig(
            format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s(): %(message)s",
            level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p")
    else:
        logging.basicConfig(
            format="%(levelname)s: %(message)s", level=loglevel)

    # load the data from the files
    if arguments.path is not None:

        # check if path for folder with data is okay
        if os.path.exists(arguments.path):

            # create new folder for output and get the path
            new_folder_path = create_folder_for_output(arguments.out)

            # open and read the main code file
            code = read_code(arguments.path)

            # get the user defined modules used in the main code
            module_imports = get_modules_and_paths(code)
            #print(module_imports)

            # get the line numbers for all imported classes from which objects are created
            module_line_numbers = get_line_numbers_class_instances(module_imports, code)
            #print(module_line_numbers)

            # find all function definitions in this code and analyze where they start and end
            function_definitions = get_function_definitions(code)
            #print(function_definitions)

            # check if a function is called inside another function in this file
            function_definitions = recursive_check_function_in_function_declarations(function_definitions)
            #print(function_definitions)

            # check inside which methods the objects are created
            module_line_numbers = get_parent_of_object_declaration(module_line_numbers, function_definitions)
            print(module_line_numbers)

            #TODO: baustelle hier...
            for i in range(len(module_line_numbers)):
                new_file_name = module_line_numbers[i][1]
                parent_function = module_line_numbers[i][3]
                new_file_name += ".py"
                print(parent_function)
                if os.path.exists(new_file_name):
                    
                    # open and read the code file
                    new_code = read_code(new_file_name)

                    # get the user defined modules used in the code
                    new_module_imports = get_modules_and_paths(new_code)
                    #print(new_module_imports)

                    # get the line numbers for all imported classes from which objects are created
                    new_module_line_numbers = get_line_numbers_class_instances(new_module_imports, new_code)
                    #print(new_module_line_numbers)

                    # find all function definitions in this code and analyze where they start and end
                    new_function_definitions = get_function_definitions(new_code)
                    #print(new_function_definitions)

                    # check if a function is called inside another function in this file
                    new_function_definitions = recursive_check_function_in_function_declarations(new_function_definitions)
                    print(new_function_definitions)

                    # check inside which methods the objects are created
                    new_module_line_numbers = get_parent_of_object_declaration(new_module_line_numbers, new_function_definitions)
                    print(new_module_line_numbers)

                    #TODO: carry over the modifiers for annotation, make sure this also works with the recursion!!!
                    # go through all the found function declarations and add the nvtx decorators
                    new_code = add_nvtx_decorators(new_code, new_function_definitions, parent_function)

                    # add the nvtx package import to the file
                    new_code = add_nvtx_import_statement(new_code)

                    # create the new filepath
                    x = arguments.path.split("/")
                    new_path = os.path.join(new_folder_path, new_file_name)
                    #print(new_path)

                    # write the instrumented code to a file
                    write_code(new_path, new_code)

                    # TODO: from this point on the further instrumentation should be done recursively...
                else:
                    pass

            #TODO: need a recursive method that goes through all files after each other byitself, carrying the previous callstack modifiers over for annotation.

            #TODO: when finished this is enough functionality, should not waste more time on that... comment all print statements...

            #TODO: create some code that shows that the training step is called in train and also adds this info to the callpaths of the decorators...


            # go through all the found function declarations and add the nvtx decorators
            code = add_nvtx_decorators(code, function_definitions, "")

            # add the nvtx package import to the file
            code = add_nvtx_import_statement(code)

            # create the new filepath
            x = arguments.path.split("/")
            file_name = x[len(x)-1]
            new_path = os.path.join(new_folder_path, file_name)

            # write the instrumented code to a file
            write_code(new_path, code)
            
        else:
            logging.error("File does not exist.")
            sys.exit(1)
    else:
        logging.error("There was no proper file path provided.")
        sys.exit(1)

if __name__ == '__main__':
    main()
