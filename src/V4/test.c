// test_acc.c
#include <stdio.h>
#include <openacc.h>

int main() {
    printf("OpenACC device count: %d\n", acc_get_num_devices(acc_device_nvidia));
    
    int n = 10;
    #pragma acc parallel loop
    for(int i = 0; i < n; i++) {
        printf("Iteration %d\n", i);
    }
    return 0;
}
