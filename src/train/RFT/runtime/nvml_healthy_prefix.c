/* Process-local NVML enumeration guard for the verified healthy prefix 0..3.
 * NCCL resolves NVML functions through dlsym, so Python patches are insufficient.
 * Opt-in only: TOOLEQA_NVML_HEALTHY_PREFIX=4 and LD_PRELOAD=this compiled library.
 * No indices/handles/health results are fabricated or remapped. Other calls go
 * to the real library; only enumeration excludes devices outside this prefix.
 * Must NOT be used in the separate physical-GPU6 tool/judge processes.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*lookup_fn)(void *, const char *);
typedef int (*count_fn)(unsigned *);
static _Atomic(count_fn) original_count;
static _Atomic(count_fn) original_count_v2;

static int limited_count(unsigned *count, _Atomic(count_fn) *slot) {
    count_fn fn = atomic_load(slot);
    if (!fn) return 999;
    int status = fn(count);
    const char *setting = getenv("TOOLEQA_NVML_HEALTHY_PREFIX");
    if (status == 0 && setting && strcmp(setting, "4") == 0 && *count > 4)
        *count = 4;
    return status;
}
static int count_guard(unsigned *count) { return limited_count(count, &original_count); }
static int count_v2_guard(unsigned *count) { return limited_count(count, &original_count_v2); }

void *dlsym(void *handle, const char *name) {
    lookup_fn lookup = (lookup_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    void *result = lookup(handle, name);
    const char *setting = getenv("TOOLEQA_NVML_HEALTHY_PREFIX");
    if (!result || !setting || strcmp(setting, "4") != 0) return result;
    if (strcmp(name, "nvmlDeviceGetCount") == 0) {
        atomic_store(&original_count, (count_fn)result);
        return (void *)count_guard;
    }
    if (strcmp(name, "nvmlDeviceGetCount_v2") == 0) {
        atomic_store(&original_count_v2, (count_fn)result);
        return (void *)count_v2_guard;
    }
    return result;
}
