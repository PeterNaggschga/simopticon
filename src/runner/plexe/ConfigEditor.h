#ifndef SIMOPTICON_CONFIGEDITOR_H
#define SIMOPTICON_CONFIGEDITOR_H

/**
 * @file
 * In this file the header of the ConfigEditor class is defined.
 */

#include "../../ComparisonFunctions.h"
#include "nlohmann/json.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

class Parameter;

using namespace std;
using json = nlohmann::json;

/**
 * A class capable of creating `.ini` files with certain options based on a complete `omnetpp.ini`.
 * @ingroup plexe
 */
class ConfigEditor {
private:
    /**
     * Path to a directory containing a complete configuration of Plexe.
     * Can be set in config.
     */
    const filesystem::path DIR;
    /**
     * Path to the `omnetpp.ini` file in #DIR.
     */
    const filesystem::path CONFIG;
    /**
     * Path to the `optResults` directory in #DIR where the simulation result files are generated.
     */
    const filesystem::path RESULTS;
    /**
     * Configuration of the controller to be simulated.
     * Can be set in config.
     */
    const json CONTROLLER;

    /**
     * Replaces the value of the given key with the given new value in the given string.
     * @param file: A string containing the contents of an `.ini` file.
     * @param option: A string representing a key in the given file.
     * @param value: The new value of the given option in the given file.
     */
    static void replaceOption(string &file, string option, const string &value);

    /**
     * Replaces the value of the given key with the given new value in the given string.
     * Basically parses the given value to string and calls #replaceOption(string &, string, const string &).
     * @param file: A string containing the contents of an `.ini` file.
     * @param option: A string representing a key in the given file.
     * @param value: The new value of the given option in the given file.
     */
    static void replaceOption(string &file, string option, long value);

    /**
     * Sets all output directories in the given file to a directory that is named after the given number and a subdirectory of #RESULTS.
     * @param file: A string containing the contents of an `.ini` file.
     * @param runNumber: The unique number of the Parameter combination.
     */
    void setResultFiles(string &file, size_t runNumber);

    /**
     * Returns the key that defines the used controller in the given `.ini` file.
     * That is necessary for backwards compatability reasons because said key changed in Plexe 3.1.
     * @param file: A string containing the contents of an `.ini` file.
     * @return A string containing the key where the used controller is defined.
     */
    static string getControllerOption(string &file);

public:
    /**
     * Creates a ConfigEditor that creates config files in the given directory for simulation of the given controller.
     * @param directory: A path to the directory containing a Plexe configuration.
     * @param controller: A json object configuring the controller to be simulated.
     */
    ConfigEditor(filesystem::path directory, json controller);

    /**
     * Copies the config at #CONFIG to a file `.tmpx.ini` where `x` is given by @a runNumber and edits the file for the purposes of the optimization.
     * Sets the values of optimized parameters, controller, result directory and some options minimizing output of Plexe.
     * @param params: The Parameter combination to be simulated.
     * @param runNumber: An unique number of the simulated Parameter combination.
     * @param repeat: Number of repetitions to be simulated.
     */
    void createConfig(const vector<shared_ptr<Parameter>> &params, size_t runNumber, unsigned int repeat);

    /**
     * Deletes the file `.tmpx.ini` from #DIR where `x` is given by @a runId.
     * @param runId: Number of the configuration file to be deleted.
     */
    void deleteConfig(size_t runId) const;

    /**
     * Returns the directory of the Plexe configuration.
     * @return The path stored in #DIR
     */
    [[nodiscard]] const filesystem::path &getDir() const;

    /**
     * Returns the path to the created config for the Parameter combination with the given number.
     * @param runId: Number of the Parameter combination.
     * @return A path to the config for the given runId.
     */
    [[nodiscard]] filesystem::path getConfigPath(size_t runId) const;

    /**
     * Returns the path to the result files generated by simulating the Parameter combination with the given number.
     * @param runId: Number of the Parameter combination.
     * @return A path to the result files for the given runId.
     */
    [[nodiscard]] filesystem::path getResultPath(size_t runId) const;

};


#endif //SIMOPTICON_CONFIGEDITOR_H
