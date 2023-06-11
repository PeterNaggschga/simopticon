#ifndef SIMOPTICON_CONSTANTHEADWAY_H
#define SIMOPTICON_CONSTANTHEADWAY_H

#include "../Pipeline.h"
#include "../../utils/PythonScript.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedMacroInspection"
#define PY_SSIZE_T_CLEAN

#include <Python.h>

#pragma clang diagnostic pop

class ConstantHeadway : public Pipeline, public PythonScript {
private:
    const unsigned int NR_THREADS;

    PyObject *secureValue(PyObject *object);

public:
    explicit ConstantHeadway(unsigned int nrThreads);

    functionValue processOutput(filesystem::path path, set<runId> experimentIds, unsigned int pipelineId) override;

    map<pair<filesystem::path, set<runId>>, functionValue>
    processOutput(const set<pair<filesystem::path, set<runId>>> &experimentResults, unsigned int pipelineId) override;

};

#endif //SIMOPTICON_CONSTANTHEADWAY_H
