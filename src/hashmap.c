#include "hashmap.h"

#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 16384
#define LOAD_FACTOR 0.75

static unsigned long hash_string(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++) != 0) {
        hash = ((hash << 5) + hash) + (unsigned char)c;
    }
    return hash;
}

static void hashmap_rehash(HashMap *map) {
    size_t new_capacity = map->capacity * 2;
    HashNode **new_buckets = calloc(new_capacity, sizeof(HashNode *));
    if (!new_buckets) {
        return;
    }

    for (size_t i = 0; i < map->capacity; ++i) {
        HashNode *node = map->buckets[i];
        while (node) {
            HashNode *next = node->next;
            unsigned long hash = hash_string(node->key) % new_capacity;
            node->next = new_buckets[hash];
            new_buckets[hash] = node;
            node = next;
        }
    }

    free(map->buckets);
    map->buckets = new_buckets;
    map->capacity = new_capacity;
}

void hashmap_init(HashMap *map, size_t capacity) {
    size_t initial_capacity = capacity == 0 ? INITIAL_CAPACITY : capacity;
    map->buckets = calloc(initial_capacity, sizeof(HashNode *));
    map->capacity = map->buckets ? initial_capacity : 0;
    map->size = 0;
}

void hashmap_free(HashMap *map) {
    if (!map || !map->buckets) {
        return;
    }

    for (size_t i = 0; i < map->capacity; ++i) {
        HashNode *node = map->buckets[i];
        while (node) {
            HashNode *next = node->next;
            free(node->key);
            free(node);
            node = next;
        }
    }

    free(map->buckets);
    map->buckets = NULL;
    map->capacity = 0;
    map->size = 0;
}

void hashmap_increment(HashMap *map, const char *key, long long delta) {
    if (!map || !map->buckets || !key) {
        return;
    }

    unsigned long hash = hash_string(key) % map->capacity;
    HashNode *node = map->buckets[hash];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->value += delta;
            return;
        }
        node = node->next;
    }

    HashNode *new_node = malloc(sizeof(HashNode));
    if (!new_node) {
        return;
    }
    new_node->key = strdup(key);
    if (!new_node->key) {
        free(new_node);
        return;
    }
    new_node->value = delta;
    new_node->next = map->buckets[hash];
    map->buckets[hash] = new_node;
    map->size++;

    if ((double)map->size / (double)map->capacity > LOAD_FACTOR) {
        hashmap_rehash(map);
    }
}

size_t hashmap_size(const HashMap *map) {
    return map ? map->size : 0;
}

KeyValue *hashmap_to_array(const HashMap *map) {
    if (!map || !map->buckets) {
        return NULL;
    }

    KeyValue *array = malloc(map->size * sizeof(KeyValue));
    if (!array) {
        return NULL;
    }

    size_t index = 0;
    for (size_t i = 0; i < map->capacity; ++i) {
        HashNode *node = map->buckets[i];
        while (node) {
            array[index].key = node->key;
            array[index].value = node->value;
            ++index;
            node = node->next;
        }
    }

    return array;
}
