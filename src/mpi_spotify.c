#include <ctype.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hashmap.h"

#define TAG_LINE_LENGTH 1
#define TAG_LINE_DATA 2
#define TAG_WORD_COUNT 3
#define TAG_ARTIST_COUNT 4

#define MAX_TOP_RESULTS 20

static void trim_newline(char *str) {
    if (!str) {
        return;
    }
    size_t len = strlen(str);
    while (len > 0 && (str[len - 1] == '\n' || str[len - 1] == '\r')) {
        str[len - 1] = '\0';
        --len;
    }
}

static void to_lowercase(char *str) {
    if (!str) {
        return;
    }
    for (size_t i = 0; str[i] != '\0'; ++i) {
        str[i] = (char)tolower((unsigned char)str[i]);
    }
}

static void sanitize_lyrics(char *lyrics) {
    if (!lyrics) {
        return;
    }
    for (size_t i = 0; lyrics[i] != '\0'; ++i) {
        unsigned char ch = (unsigned char)lyrics[i];
        if (!isalnum(ch) && ch != '\'' && ch != ' ') {
            lyrics[i] = ' ';
        }
    }
}

static bool parse_csv_line(const char *line, char **artist_out, char **lyrics_out) {
    if (!line || !artist_out || !lyrics_out) {
        return false;
    }

    size_t length = strlen(line);
    char *buffer = malloc(length + 1);
    if (!buffer) {
        return false;
    }
    strcpy(buffer, line);

    char *fields[4] = {0};
    size_t field_index = 0;
    char *token = malloc(length + 1);
    if (!token) {
        free(buffer);
        return false;
    }

    size_t token_index = 0;
    bool in_quotes = false;
    for (size_t i = 0; i <= length; ++i) {
        char ch = buffer[i];
        bool is_end = (ch == '\0');
        if (ch == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if ((ch == ',' && !in_quotes) || is_end) {
            token[token_index] = '\0';
            fields[field_index++] = strdup(token);
            token_index = 0;
            if (field_index >= 4) {
                break;
            }
            continue;
        }
        if (ch == '\r' || ch == '\n') {
            continue;
        }
        token[token_index++] = ch;
    }

    free(buffer);
    free(token);

    if (field_index < 4) {
        for (size_t i = 0; i < field_index; ++i) {
            free(fields[i]);
        }
        return false;
    }

    *artist_out = fields[0];
    *lyrics_out = fields[3];
    free(fields[1]);
    free(fields[2]);
    return true;
}

static void free_parsed_fields(char *artist, char *lyrics) {
    free(artist);
    free(lyrics);
}

static void process_lyrics(const char *lyrics, HashMap *word_map) {
    if (!lyrics) {
        return;
    }
    char *normalized = strdup(lyrics);
    if (!normalized) {
        return;
    }
    to_lowercase(normalized);
    sanitize_lyrics(normalized);

    char *save_ptr = NULL;
    char *token = strtok_r(normalized, " ", &save_ptr);
    while (token) {
        if (strlen(token) > 0) {
            hashmap_increment(word_map, token, 1);
        }
        token = strtok_r(NULL, " ", &save_ptr);
    }

    free(normalized);
}

static void process_line(const char *line, HashMap *word_map, HashMap *artist_map) {
    char *artist = NULL;
    char *lyrics = NULL;
    if (!parse_csv_line(line, &artist, &lyrics)) {
        return;
    }

    trim_newline(artist);
    trim_newline(lyrics);
    to_lowercase(artist);

    if (artist && strlen(artist) > 0) {
        hashmap_increment(artist_map, artist, 1);
    }

    process_lyrics(lyrics, word_map);
    free_parsed_fields(artist, lyrics);
}

static void send_map(HashMap *map, int dest_rank, int tag_base) {
    int entry_count = (int)hashmap_size(map);
    MPI_Send(&entry_count, 1, MPI_INT, dest_rank, tag_base, MPI_COMM_WORLD);
    if (entry_count == 0) {
        return;
    }

    KeyValue *entries = hashmap_to_array(map);
    if (!entries) {
        return;
    }

    for (int i = 0; i < entry_count; ++i) {
        int key_length = (int)strlen(entries[i].key);
        MPI_Send(&key_length, 1, MPI_INT, dest_rank, tag_base + 1, MPI_COMM_WORLD);
        MPI_Send(entries[i].key, key_length, MPI_CHAR, dest_rank, tag_base + 2, MPI_COMM_WORLD);
        MPI_Send(&entries[i].value, 1, MPI_LONG_LONG, dest_rank, tag_base + 3, MPI_COMM_WORLD);
    }

    free(entries);
}

static void receive_map(HashMap *map, int source_rank, int tag_base) {
    int entry_count = 0;
    MPI_Recv(&entry_count, 1, MPI_INT, source_rank, tag_base, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < entry_count; ++i) {
        int key_length = 0;
        MPI_Recv(&key_length, 1, MPI_INT, source_rank, tag_base + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char *key = malloc((size_t)key_length + 1);
        if (!key) {
            char *discard = malloc((size_t)key_length);
            MPI_Recv(discard, key_length, MPI_CHAR, source_rank, tag_base + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            free(discard);
            long long value;
            MPI_Recv(&value, 1, MPI_LONG_LONG, source_rank, tag_base + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
        }
        MPI_Recv(key, key_length, MPI_CHAR, source_rank, tag_base + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        key[key_length] = '\0';
        long long value = 0;
        MPI_Recv(&value, 1, MPI_LONG_LONG, source_rank, tag_base + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        hashmap_increment(map, key, value);
        free(key);
    }
}

static int compare_desc(const void *a, const void *b) {
    const KeyValue *left = (const KeyValue *)a;
    const KeyValue *right = (const KeyValue *)b;
    if (right->value > left->value) {
        return 1;
    }
    if (right->value < left->value) {
        return -1;
    }
    return strcmp(left->key, right->key);
}

static void print_top_results(const char *title, HashMap *map) {
    size_t size = hashmap_size(map);
    printf("\n%s (total %zu entradas)\n", title, size);
    printf("----------------------------------------\n");
    if (size == 0) {
        return;
    }

    KeyValue *entries = hashmap_to_array(map);
    if (!entries) {
        return;
    }
    qsort(entries, size, sizeof(KeyValue), compare_desc);
    size_t limit = size < MAX_TOP_RESULTS ? size : MAX_TOP_RESULTS;
    for (size_t i = 0; i < limit; ++i) {
        printf("%-30s %10lld\n", entries[i].key, entries[i].value);
    }
    free(entries);
}

static void broadcast_failure(int world_size) {
    int terminate = -1;
    for (int rank = 1; rank < world_size; ++rank) {
        MPI_Send(&terminate, 1, MPI_INT, rank, TAG_LINE_LENGTH, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "Uso: mpirun -np <processos> ./mpi_spotify <caminho_csv>\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *csv_path = argv[1];

    HashMap word_map;
    HashMap artist_map;
    hashmap_init(&word_map, 0);
    hashmap_init(&artist_map, 0);

    double start_time = MPI_Wtime();

    if (world_rank == 0) {
        FILE *file = fopen(csv_path, "r");
        if (!file) {
            perror("Erro ao abrir arquivo CSV");
            broadcast_failure(world_size);
            hashmap_free(&word_map);
            hashmap_free(&artist_map);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        char *line = NULL;
        size_t len = 0;
        ssize_t read = getline(&line, &len, file); // header
        (void)read;

        long long line_index = 0;
        while ((read = getline(&line, &len, file)) != -1) {
            int target_rank = (int)(line_index % world_size);
            if (target_rank == 0) {
                process_line(line, &word_map, &artist_map);
            } else {
                int length = (int)strlen(line);
                MPI_Send(&length, 1, MPI_INT, target_rank, TAG_LINE_LENGTH, MPI_COMM_WORLD);
                MPI_Send(line, length, MPI_CHAR, target_rank, TAG_LINE_DATA, MPI_COMM_WORLD);
            }
            ++line_index;
        }

        free(line);
        fclose(file);

        int terminate = -1;
        for (int rank = 1; rank < world_size; ++rank) {
            MPI_Send(&terminate, 1, MPI_INT, rank, TAG_LINE_LENGTH, MPI_COMM_WORLD);
        }
    } else {
        while (1) {
            int length = 0;
            MPI_Recv(&length, 1, MPI_INT, 0, TAG_LINE_LENGTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (length == -1) {
                break;
            }
            char *line = malloc((size_t)length + 1);
            if (!line) {
                char *discard = malloc((size_t)length);
                MPI_Recv(discard, length, MPI_CHAR, 0, TAG_LINE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                free(discard);
                continue;
            }
            MPI_Recv(line, length, MPI_CHAR, 0, TAG_LINE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            line[length] = '\0';
            process_line(line, &word_map, &artist_map);
            free(line);
        }
    }

    double processing_time = MPI_Wtime() - start_time;

    if (world_rank == 0) {
        for (int source = 1; source < world_size; ++source) {
            receive_map(&word_map, source, TAG_WORD_COUNT);
            receive_map(&artist_map, source, TAG_ARTIST_COUNT);
        }
    } else {
        send_map(&word_map, 0, TAG_WORD_COUNT);
        send_map(&artist_map, 0, TAG_ARTIST_COUNT);
    }

    double total_time = MPI_Wtime() - start_time;

    double max_processing_time = 0.0;
    double min_processing_time = 0.0;
    double avg_processing_time = 0.0;
    MPI_Reduce(&processing_time, &max_processing_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&processing_time, &min_processing_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&processing_time, &avg_processing_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_total_time = 0.0;
    double min_total_time = 0.0;
    double avg_total_time = 0.0;
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &min_total_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &avg_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        avg_processing_time /= world_size;
        avg_total_time /= world_size;

        print_top_results("Palavras mais frequentes", &word_map);
        print_top_results("Artistas com mais músicas", &artist_map);

        printf("\nMétricas de desempenho:\n");
        printf("Tempo de processamento - máximo: %.3f s | mínimo: %.3f s | médio: %.3f s\n",
               max_processing_time, min_processing_time, avg_processing_time);
        printf("Tempo total - máximo: %.3f s | mínimo: %.3f s | médio: %.3f s\n",
               max_total_time, min_total_time, avg_total_time);
    }

    hashmap_free(&word_map);
    hashmap_free(&artist_map);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
