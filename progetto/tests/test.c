/**
 * \file main.c
 * \brief Example test program for OBDD with 10 variables (x0..x9)
 *
 * This file demonstrates how to create more "sophisticated" BDDs for:
 *    f0(x0,x1) = x0 AND x1
 *    f1(x2,x4) = x2 OR x4
 * then we combine them with XOR, etc. We do multiple tests, both fixed and random
 * assignments of (x0..x9).
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>   // for random
#include "obdd.h"

/**
 * Helper function: build BDD for "x0 AND x1"
 * We'll ignore the other variables x2..x9 in this BDD.
 */
static OBDDNode* build_f0_x0_and_x1() {
    // costruiamo un nodo per x0
    // - se x0=0 => restituisce 0
    // - se x0=1 => dobbiamo guardare x1
    OBDDNode* zeroLeaf = obdd_constant(0);
    OBDDNode* oneLeaf  = obdd_constant(1);

    // BDD per x1 => se x1=0 => 0, se x1=1 => 1
    // varIndex=1
    OBDDNode* nodeForX1 = obdd_node_create(
        1,
        zeroLeaf,  // x1=0 => 0
        oneLeaf    // x1=1 => 1
    );

    // BDD per x0 => se x0=0 => 0, se x0=1 => nodeForX1
    // varIndex=0
    OBDDNode* root = obdd_node_create(
        0,
        zeroLeaf,      // x0=0 => 0
        nodeForX1      // x0=1 => poi valuta x1
    );
    return root;
}

/**
 * Helper function: build BDD for "x2 OR x4"
 * We'll ignore x0,x1,x3,ecc. in questa definizione, e useremo varIndex=2,4
 */
static OBDDNode* build_f1_x2_or_x4() {
    OBDDNode* zeroLeaf = obdd_constant(0);
    OBDDNode* oneLeaf  = obdd_constant(1);

    // costruiamo un nodo per x2 => se x2=0 => ? , se x2=1 => 1
    // se x2=1 => la funzione = 1 (OR => se x2=1 => vince)
    // se x2=0 => dobbiamo guardare x4
    // Quindi:
    // varIndex=2
    //   lowChild => un nodo che dipende da x4
    //   highChild => 1
    OBDDNode* nodeForX4 = obdd_node_create(
        4,             // varIndex=4
        zeroLeaf,      // x4=0 => 0
        oneLeaf        // x4=1 => 1
    );

    OBDDNode* nodeForX2 = obdd_node_create(
        2,
        nodeForX4,     // x2=0 => poi dipende da x4
        oneLeaf        // x2=1 => subito 1
    );

    return nodeForX2;
}

int main(void) {
    // 1. Abbiamo 10 variabili: x0..x9
    int varOrder[10] = {0,1,2,3,4,5,6,7,8,9};

    // 2. Creiamo un OBDD per f0(x0,x1) e uno per f1(x2,x4)
    //    NB: Ognuno avrà numVars=10, ma fisicamente userà 2 di quelle varIndex
    OBDD* bddF0 = obdd_create(10, varOrder);  // x0 AND x1
    OBDD* bddF1 = obdd_create(10, varOrder);  // x2 OR x4

    bddF0->root = build_f0_x0_and_x1();
    bddF1->root = build_f1_x2_or_x4();

    // 3. Facciamo un test di base: f0 e f1 su un assegnamento a mano
    int assignmentA[10] = {1,1,1,0,0,0,0,0,0,0};
    // x0=1, x1=1 => f0=1, x2=1, x4=0 => f1=1 => ...
    int valF0A = obdd_evaluate(bddF0, assignmentA);
    int valF1A = obdd_evaluate(bddF1, assignmentA);
    printf("Test #1: x0=1,x1=1,x2=1,x4=0 => f0=%d, f1=%d\n", valF0A, valF1A);

    // 4. Facciamo un AND, OR, XOR tra f0 e f1
    // f0 AND f1
    OBDDNode* rootAnd = obdd_apply(bddF0, bddF1, 0); // 0=AND
    // f0 OR f1
    OBDDNode* rootOr  = obdd_apply(bddF0, bddF1, 1); // 1=OR
    // f0 XOR f1
    OBDDNode* rootXor = obdd_apply(bddF0, bddF1, 3); // 3=XOR

    // 5. Valutiamo x0=1,x1=1,x2=1,x4=0 su f0 AND f1, f0 OR f1, f0 XOR f1
    int valAND = obdd_evaluate(&(OBDD){.root=rootAnd, .numVars=10, .varOrder=varOrder}, assignmentA);
    int valOR  = obdd_evaluate(&(OBDD){.root=rootOr,  .numVars=10, .varOrder=varOrder}, assignmentA);
    int valXOR = obdd_evaluate(&(OBDD){.root=rootXor, .numVars=10, .varOrder=varOrder}, assignmentA);
    printf("f0 AND f1 => %d, f0 OR f1 => %d, f0 XOR f1 => %d\n", valAND, valOR, valXOR);

    // 6. Facciamo un NOT su f1, e testiamolo con un assegnamento differente
    //    useremo la convenzione: obdd_apply(bddF1, NULL, 2) => NOT(bddF1)
    OBDDNode* rootNotF1 = obdd_apply(bddF1, NULL, 2);

    int assignmentB[10] = {0,0,1,0,1, 0,0,0,0,0};
    // x0=0, x1=0 => f0=0, x2=1, x4=1 => f1=1 => NOT(f1)=0
    int valNotF1 = obdd_evaluate(&(OBDD){.root=rootNotF1, .numVars=10, .varOrder=varOrder}, assignmentB);
    printf("Con x0=0,x1=0,x2=1,x4=1 => f1=1 => NOT(f1)=%d (atteso 0)\n", valNotF1);

    // 7. Test multiplo random
    //    Proviamo 5 assegnamenti casuali per vedere i valori di f0 e f1
    srand((unsigned)time(NULL));
    printf("\n--- Test random su 5 assignment ---\n");
    for (int i=0; i<5; i++){
        int assignmentR[10];
        // generiamo 0/1 random
        for (int j=0; j<10; j++){
            assignmentR[j] = rand() % 2;
        }
        // calcoliamo
        int valF0 = obdd_evaluate(bddF0, assignmentR);
        int valF1 = obdd_evaluate(bddF1, assignmentR);

        printf("RANDOM #%d: x0=%d x1=%d x2=%d x4=%d => f0=%d, f1=%d\n", i,
               assignmentR[0], assignmentR[1], assignmentR[2], assignmentR[4],
               valF0, valF1);
    }

    // 8. Esempio di riduzione su rootXor (anche se fa poco)
    OBDDNode* reducedXor = obdd_reduce(rootXor);
    (void)reducedXor;

    // 9. Distruggiamo i BDD
    obdd_destroy(bddF0);
    obdd_destroy(bddF1);

    printf("\n[SYSTEM] Test completati con successo.\n");
    return 0;
}
