/**
 * \file main.c
 * \brief Example test program for OBDD with 10 variables
 *
 * This file demonstrates how to create a small OBDD for x0, another for x9,
 * apply logical operations (AND, OR, NOT), and evaluate the results on
 * a 10-variable assignment. The rest of x1..x8 are unused for simplicity,
 * but we show how you'd handle 10 variables total.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "obdd.h"

int main(void) {
    // 1. Array di 10 variabili: x0, x1, x2, ..., x9
    int varOrder[10] = {0,1,2,3,4,5,6,7,8,9};

    // 2. Creiamo due OBDD, entrambi hanno un'ordinazione su 10 variabili
    OBDD* bdd1 = obdd_create(10, varOrder);
    OBDD* bdd2 = obdd_create(10, varOrder);

    // 3. Assegniamo a bdd1 la funzione f1(x0,...,x9) = x0
    //    -> un nodo con varIndex=0, lowChild= false, highChild= true
    OBDDNode* node1 = obdd_node_create(
        0,                 // varIndex = x0
        obdd_constant(0),  // se x0=0 => false
        obdd_constant(1)   // se x0=1 => true
    );
    bdd1->root = node1;

    // 4. Assegniamo a bdd2 la funzione f2(x0,...,x9) = x9
    //    -> un nodo con varIndex=9, lowChild= false, highChild= true
    OBDDNode* node2 = obdd_node_create(
        9,                 // varIndex = x9
        obdd_constant(0),  // se x9=0 => false
        obdd_constant(1)   // se x9=1 => true
    );
    bdd2->root = node2;

    // 5. Valutiamo bdd1 (cioè x0) con x0=0 e TUTTE le altre variabili =0
    //    Quindi creiamo un assignment[10]
    int assign1[10] = {0,0,0,0,0,0,0,0,0,0};
    int val1 = obdd_evaluate(bdd1, assign1);
    printf("Valore di x0 con x0=0 => %d (atteso 0)\n", val1);

    // 6. Creiamo un assignment dove x0=1 e x9=1, e il resto=0
    int assign2[10] = {1,0,0,0,0,0,0,0,0,1};

    // 7. Eseguiamo x0 AND x9 e valutiamo su quell'assegnamento
    OBDDNode* andRoot = obdd_apply(bdd1, bdd2, 0 /* 0=AND */);
    int valAND = obdd_evaluate(&(OBDD){.root=andRoot, .numVars=10, .varOrder=varOrder}, assign2);
    printf("Valore di (x0 AND x9) con x0=1 e x9=1 => %d (atteso 1)\n", valAND);

    // 8. Eseguiamo x0 OR x9 e valutiamo su x0=1, x9=1
    OBDDNode* orRoot = obdd_apply(bdd1, bdd2, 1 /* 1=OR */);
    int valOR = obdd_evaluate(&(OBDD){.root=orRoot, .numVars=10, .varOrder=varOrder}, assign2);
    printf("Valore di (x0 OR x9) con x0=1 e x9=1 => %d (atteso 1)\n", valOR);

    // 9. Testiamo NOT su bdd2 (cioè NOT x9)
    //    useremo la convenzione op=2 => NOT
    //    passiamo bdd2 e bdd2NULL? -> In questo codice, passiamo bdd2 e bdd2==NULL ...
    //    Nel Tuo obdd, se fai NOT(bdd1) => bdd2=NULL, op=2
    OBDDNode* notRoot = obdd_apply(bdd2, NULL, 2 /* NOT */);
    // Valutiamo NOT(x9) su x0=1, x9=1 => atteso 0
    int valNOT = obdd_evaluate(&(OBDD){.root=notRoot, .numVars=10, .varOrder=varOrder}, assign2);
    printf("Valore di NOT(x9) con x9=1 => %d (atteso 0)\n", valNOT);

    // 10. Esempio di riduzione (fa nulla, ma la chiamiamo)
    OBDDNode* reducedAND = obdd_reduce(andRoot);
    (void)reducedAND; // solo esempio

    // 11. Distruggiamo i BDD (anche se non stiamo ricorsivamente free-ando i nodi)
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);

    printf("\n[Tutte le valutazioni sono state eseguite con successo!]\n");
    return 0;
}
