#ifndef HASHMAP_H
#define HASHMAP_H

#include <stddef.h>

typedef struct HashNode {
    char *key;
    long long value;
    struct HashNode *next;
} HashNode;

typedef struct {
    HashNode **buckets;
    size_t capacity;
    size_t size;
} HashMap;

typedef struct {
    const char *key;
    long long value;
} KeyValue;

void hashmap_init(HashMap *map, size_t capacity);
void hashmap_free(HashMap *map);
void hashmap_increment(HashMap *map, const char *key, long long delta);
size_t hashmap_size(const HashMap *map);
KeyValue *hashmap_to_array(const HashMap *map);

#endif
