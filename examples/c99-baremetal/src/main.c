/* ===========================================================================
 * embedded-nn C99 Bare-Metal Deployment Demo
 *
 * Demonstrates compiling and executing an embedded-nn generated model in pure C99
 * with zero dynamic memory allocation and zero external library dependencies.
 * =========================================================================== */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "gesture_model.h"

// Statically allocated buffers (Zero heap / malloc calls!)
static uint8_t g_arena[GESTUREMODEL_ARENA_SIZE_BYTES];
static int8_t  g_input[GESTUREMODEL_INPUT_DIM];
static int8_t  g_output[GESTUREMODEL_OUTPUT_DIM];

int main(void) {
    printf("=== embedded-nn C99 Bare-Metal Inference Demo ===\n");
    printf("Model:            GestureModel\n");
    printf("Input Dimension:  %d bytes (INT8)\n", GESTUREMODEL_INPUT_DIM);
    printf("Output Dimension: %d bytes (INT8)\n", GESTUREMODEL_OUTPUT_DIM);
    printf("SRAM Arena Size:  %d bytes\n\n", GESTUREMODEL_ARENA_SIZE_BYTES);

    // Test Case 1: Positive gesture feature vector
    printf("[1] Running Inference on Positive Input Vector...\n");
    for (int i = 0; i < GESTUREMODEL_INPUT_DIM; i++) {
        g_input[i] = 10;
    }

    int status = gesturemodel_predict(g_input, g_output, g_arena);
    if (status != 0) {
        printf("Error: Model inference returned code %d\n", status);
        return status;
    }

    printf("  Input:   [10, 10, 10, ...]\n");
    printf("  Output:  Logit[0] = %d, Logit[1] = %d\n", g_output[0], g_output[1]);
    printf("  Top Class: %s (ID %d)\n\n", (g_output[0] > g_output[1]) ? "Class 0" : "Class 1", (g_output[0] > g_output[1]) ? 0 : 1);

    // Test Case 2: Negative gesture feature vector
    printf("[2] Running Inference on Negative Input Vector...\n");
    for (int i = 0; i < GESTUREMODEL_INPUT_DIM; i++) {
        g_input[i] = -10;
    }

    status = gesturemodel_predict(g_input, g_output, g_arena);
    if (status != 0) {
        printf("Error: Model inference returned code %d\n", status);
        return status;
    }

    printf("  Input:   [-10, -10, -10, ...]\n");
    printf("  Output:  Logit[0] = %d, Logit[1] = %d\n", g_output[0], g_output[1]);
    printf("  Top Class: %s (ID %d)\n\n", (g_output[0] > g_output[1]) ? "Class 0" : "Class 1", (g_output[0] > g_output[1]) ? 0 : 1);

    printf("C99 Inference successfully completed with 0 heap bytes allocated!\n");
    return 0;
}
