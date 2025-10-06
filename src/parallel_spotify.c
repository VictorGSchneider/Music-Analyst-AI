/*
 * Aplicação principal em MPI para análise do dataset Spotify Million Song.
 *
 * O programa divide o arquivo CSV entre todos os processos e realiza três
 * tarefas principais de forma paralela: contagem de palavras nas letras,
 * contagem de músicas por artista e consolidação de métricas de tempo. O
 * processo de rank mestre agrega os resultados parciais, gera arquivos de
 * saída e salva medições de desempenho para posterior análise.
 *
 * A documentação está escrita em português para facilitar a consulta pelos
 * avaliadores do trabalho.
 */
#define _GNU_SOURCE
#include <mpi.h>
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <unistd.h>
#define MKDIR(path) mkdir(path, 0777)
#endif

#define DEFAULT_WORD_LIMIT 100
#define DEFAULT_ARTIST_LIMIT 50

typedef long long CountType;

/* Estrutura chave-valor usada para armazenar entradas de tabelas de hash. */
typedef struct {
    char *key;
    CountType value;
} Entry;

/*
 * Implementação simples de tabela de hash com endereçamento aberto, usada
 * tanto para a contagem de palavras quanto para a contagem de artistas.
 */
typedef struct {
    Entry *entries;
    size_t capacity;
    size_t size;
} HashTable;

/* Retorna a próxima potência de dois maior ou igual ao valor solicitado. */
static size_t next_power_of_two(size_t value) {
    size_t power = 1;
    while (power < value) {
        power <<= 1U;
    }
    return power < 8 ? 8 : power;
}

/* Calcula o hash FNV-1a para strings, garantindo boa distribuição. */
static uint64_t hash_string(const char *str) {
    const uint64_t fnv_prime = 1099511628211ULL;
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)str; *p; ++p) {
        hash ^= (uint64_t)(*p);
        hash *= fnv_prime;
    }
    return hash;
}

/* Inicializa a tabela de hash com a capacidade solicitada. */
static void ht_init(HashTable *ht, size_t initial_capacity) {
    ht->capacity = next_power_of_two(initial_capacity);
    ht->size = 0;
    ht->entries = (Entry *)calloc(ht->capacity, sizeof(Entry));
    if (!ht->entries) {
        fprintf(stderr, "Failed to allocate hash table with capacity %zu\n", ht->capacity);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

/* Libera os recursos associados à tabela de hash. */
static void ht_free(HashTable *ht) {
    if (!ht || !ht->entries) {
        return;
    }
    for (size_t i = 0; i < ht->capacity; ++i) {
        if (ht->entries[i].key) {
            free(ht->entries[i].key);
        }
    }
    free(ht->entries);
    ht->entries = NULL;
    ht->capacity = 0;
    ht->size = 0;
}

/* Duplica a tabela de hash quando o fator de carga fica elevado. */
static void ht_resize(HashTable *ht, size_t new_capacity) {
    HashTable resized;
    ht_init(&resized, new_capacity);
    for (size_t i = 0; i < ht->capacity; ++i) {
        if (ht->entries[i].key) {
            Entry *entry = &ht->entries[i];
            const size_t mask = resized.capacity - 1U;
            size_t index = hash_string(entry->key) & mask;
            while (resized.entries[index].key) {
                index = (index + 1U) & mask;
            }
            resized.entries[index].key = strdup(entry->key);
            if (!resized.entries[index].key) {
                fprintf(stderr, "Failed to duplicate key during resize\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            resized.entries[index].value = entry->value;
            resized.size++;
        }
    }
    ht_free(ht);
    *ht = resized;
}

/* Insere ou atualiza uma chave na tabela de hash. */
static void ht_put(HashTable *ht, const char *key, CountType delta) {
    if (delta == 0) {
        return;
    }
    if ((double)ht->size / (double)ht->capacity > 0.7) {
        ht_resize(ht, ht->capacity << 1U);
    }
    const size_t mask = ht->capacity - 1U;
    size_t index = hash_string(key) & mask;
    while (ht->entries[index].key) {
        if (strcmp(ht->entries[index].key, key) == 0) {
            ht->entries[index].value += delta;
            return;
        }
        index = (index + 1U) & mask;
    }
    ht->entries[index].key = strdup(key);
    if (!ht->entries[index].key) {
        fprintf(stderr, "Failed to duplicate key '%s'\n", key);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ht->entries[index].value = delta;
    ht->size++;
}

/* Mescla todas as entradas de uma tabela de hash em outra. */
static void ht_merge(HashTable *dest, const HashTable *src) {
    for (size_t i = 0; i < src->capacity; ++i) {
        if (src->entries[i].key) {
            ht_put(dest, src->entries[i].key, src->entries[i].value);
        }
    }
}

/* Converte o conteúdo da tabela para um vetor denso de entradas. */
static Entry *ht_to_array(const HashTable *ht, size_t *out_size) {
    Entry *array = (Entry *)malloc(sizeof(Entry) * ht->size);
    if (!array) {
        fprintf(stderr, "Failed to allocate array for hash table export\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    size_t idx = 0;
    for (size_t i = 0; i < ht->capacity; ++i) {
        if (ht->entries[i].key) {
            array[idx++] = ht->entries[i];
        }
    }
    *out_size = idx;
    return array;
}

/* Função de ordenação: valores maiores primeiro e, em empate, ordem alfabética. */
static int entry_compare_desc(const void *a, const void *b) {
    const Entry *ea = (const Entry *)a;
    const Entry *eb = (const Entry *)b;
    if (ea->value < eb->value) {
        return 1;
    }
    if (ea->value > eb->value) {
        return -1;
    }
    return strcmp(ea->key, eb->key);
}

/* Remove espaços em branco no início e fim da string modificando-a in place. */
static void trim_inplace(char *value) {
    if (!value) {
        return;
    }
    size_t len = strlen(value);
    size_t start = 0;
    while (start < len && isspace((unsigned char)value[start])) {
        start++;
    }
    size_t end = len;
    while (end > start && isspace((unsigned char)value[end - 1])) {
        end--;
    }
    if (start > 0) {
        memmove(value, value + start, end - start);
    }
    value[end - start] = '\0';
}

/*
 * Remove aspas e escapes de um campo CSV, retornando uma nova string limpa.
 * O chamador é responsável por liberar o ponteiro retornado.
 */
static char *clean_field(const char *field) {
    size_t len = strlen(field);
    size_t start = 0;
    size_t end = len;
    while (start < len && isspace((unsigned char)field[start])) {
        start++;
    }
    while (end > start && isspace((unsigned char)field[end - 1])) {
        end--;
    }
    int quoted = (end > start + 1 && field[start] == '"' && field[end - 1] == '"');
    if (quoted) {
        start++;
        end--;
    }
    char *result = (char *)malloc(end - start + 1);
    if (!result) {
        fprintf(stderr, "Failed to allocate memory for CSV field\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    size_t j = 0;
    for (size_t i = start; i < end; ++i) {
        if (field[i] == '"' && i + 1 < end && field[i + 1] == '"') {
            result[j++] = '"';
            i++;
        } else {
            result[j++] = field[i];
        }
    }
    result[j] = '\0';
    trim_inplace(result);
    return result;
}

/* Extrai artista e letra a partir de uma linha CSV, já descontando o cabeçalho. */
static int parse_csv_line(const char *line, char **artist_out, char **lyrics_out) {
    if (!line || !artist_out || !lyrics_out) {
        return 0;
    }
    char *buffer = strdup(line);
    if (!buffer) {
        return 0;
    }
    size_t len = strlen(buffer);
    while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
        buffer[--len] = '\0';
    }
    char *fields[4] = {0};
    int field_index = 0;
    int in_quotes = 0;
    char *ptr = buffer;
    char *token_start = buffer;
    while (*ptr) {
        if (*ptr == '"') {
            if (in_quotes && ptr[1] == '"') {
                ptr++;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (*ptr == ',' && !in_quotes) {
            *ptr = '\0';
            if (field_index < 4) {
                fields[field_index++] = token_start;
            }
            token_start = ptr + 1;
            if (field_index == 3) {
                break;
            }
        }
        ptr++;
    }
    if (field_index < 3) {
        free(buffer);
        return 0;
    }
    fields[3] = token_start;
    *artist_out = clean_field(fields[0]);
    *lyrics_out = clean_field(fields[3]);
    free(buffer);
    return *artist_out && *lyrics_out;
}

/* Escreve uma linha CSV escapando aspas para o campo textual. */
static void write_csv_entry(FILE *fp, const char *key, CountType value) {
    fputc('"', fp);
    for (const char *p = key; *p; ++p) {
        if (*p == '"') {
            fputc('"', fp);
            fputc('"', fp);
        } else {
            fputc(*p, fp);
        }
    }
    fputc('"', fp);
    fprintf(fp, ",%lld\n", value);
}

/*
 * Exporta os resultados agregados para um arquivo CSV, ordenando os itens
 * pelos maiores valores e respeitando o limite solicitado.
 */
static void write_table_csv(const HashTable *ht, const char *filepath, const char *key_header, int limit) {
    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file %s: %s\n", filepath, strerror(errno));
        return;
    }
    fprintf(fp, "%s,count\n", key_header);
    size_t array_size = 0;
    Entry *entries = ht_to_array(ht, &array_size);
    qsort(entries, array_size, sizeof(Entry), entry_compare_desc);
    size_t max_items = array_size;
    if (limit > 0 && (size_t)limit < array_size) {
        max_items = (size_t)limit;
    }
    for (size_t i = 0; i < max_items; ++i) {
        write_csv_entry(fp, entries[i].key, entries[i].value);
    }
    free(entries);
    fclose(fp);
}

/* Tokeniza as letras, acumula contagem por palavra e atualiza o total geral. */
static void process_lyrics(HashTable *word_counts, const char *lyrics, CountType *total_words) {
    size_t capacity = 64;
    char *buffer = (char *)malloc(capacity);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer for tokenisation\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    size_t length = 0;
    for (const unsigned char *p = (const unsigned char *)lyrics; *p; ++p) {
        if (isalnum(*p)) {
            if (length + 1 >= capacity) {
                capacity *= 2U;
                char *tmp = (char *)realloc(buffer, capacity);
                if (!tmp) {
                    free(buffer);
                    fprintf(stderr, "Failed to grow token buffer\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                buffer = tmp;
            }
            buffer[length++] = (char)tolower(*p);
        } else {
            if (length > 0) {
                buffer[length] = '\0';
                ht_put(word_counts, buffer, 1);
                (*total_words)++;
                length = 0;
            }
        }
    }
    if (length > 0) {
        buffer[length] = '\0';
        ht_put(word_counts, buffer, 1);
        (*total_words)++;
    }
    free(buffer);
}

/* Envia todas as entradas de uma tabela de hash para outro processo MPI. */
static void send_hash_table(const HashTable *ht, int dest, int tag_base, MPI_Comm comm) {
    int entry_count = (int)ht->size;
    MPI_Send(&entry_count, 1, MPI_INT, dest, tag_base, comm);
    for (size_t i = 0; i < ht->capacity; ++i) {
        if (!ht->entries[i].key) {
            continue;
        }
        int len = (int)strlen(ht->entries[i].key);
        CountType value = ht->entries[i].value;
        MPI_Send(&len, 1, MPI_INT, dest, tag_base + 1, comm);
        MPI_Send(ht->entries[i].key, len, MPI_CHAR, dest, tag_base + 2, comm);
        MPI_Send(&value, 1, MPI_LONG_LONG, dest, tag_base + 3, comm);
    }
}

/* Recebe uma tabela de hash serializada e mescla os valores no destino. */
static void receive_hash_table(HashTable *dest, int source, int tag_base, MPI_Comm comm) {
    MPI_Status status;
    int entry_count = 0;
    MPI_Recv(&entry_count, 1, MPI_INT, source, tag_base, comm, &status);
    for (int i = 0; i < entry_count; ++i) {
        int len = 0;
        MPI_Recv(&len, 1, MPI_INT, source, tag_base + 1, comm, &status);
        char *buffer = (char *)malloc(len + 1);
        if (!buffer) {
            fprintf(stderr, "Failed to allocate buffer for received key\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        MPI_Recv(buffer, len, MPI_CHAR, source, tag_base + 2, comm, &status);
        buffer[len] = '\0';
        CountType value = 0;
        MPI_Recv(&value, 1, MPI_LONG_LONG, source, tag_base + 3, comm, &status);
        ht_put(dest, buffer, value);
        free(buffer);
    }
}

/* Obtém o tamanho do arquivo de entrada em bytes. */
static long long get_file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    return (long long)st.st_size;
}

/* Mede o comprimento da linha de cabeçalho para permitir o fatiamento do CSV. */
static long long compute_header_length(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        return -1;
    }
    char *line = NULL;
    size_t n = 0;
    ssize_t read = getline(&line, &n, fp);
    long long header_len = -1;
    if (read >= 0) {
        header_len = (long long)ftell(fp);
    }
    free(line);
    fclose(fp);
    return header_len;
}

/* Cria o diretório de saída caso ele não exista. */
static void ensure_output_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return;
    }
    if (MKDIR(path) != 0 && errno != EEXIST) {
        fprintf(stderr, "Failed to create directory %s: %s\n", path, strerror(errno));
    }
}

/* Função principal que distribui o trabalho entre os processos MPI. */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <n> %s <dataset.csv> [--word-limit N] [--artist-limit N] [--output-dir DIR]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *dataset_path = argv[1];
    int word_limit = DEFAULT_WORD_LIMIT;
    int artist_limit = DEFAULT_ARTIST_LIMIT;
    const char *output_dir = "output";
    char word_output_path[512] = {0};
    char artist_output_path[512] = {0};
    char metrics_output_path[512] = {0};

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--word-limit") == 0 && i + 1 < argc) {
            word_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--artist-limit") == 0 && i + 1 < argc) {
            artist_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (rank == 0) {
            fprintf(stderr, "Ignoring unknown argument: %s\n", argv[i]);
        }
    }

    long long header_len = 0;
    long long file_size = 0;
    if (rank == 0) {
        header_len = compute_header_length(dataset_path);
        file_size = get_file_size(dataset_path);
        if (header_len < 0 || file_size < 0) {
            fprintf(stderr, "Failed to obtain dataset metadata\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&header_len, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (file_size <= header_len) {
        if (rank == 0) {
            fprintf(stderr, "Dataset appears to be empty\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    long long data_bytes = file_size - header_len;
    long long base_chunk = data_bytes / world_size;
    long long remainder = data_bytes % world_size;
    long long local_start = header_len + rank * base_chunk + (rank < remainder ? rank : remainder);
    long long local_end = local_start + base_chunk + (rank < remainder ? 1 : 0);
    if (rank == world_size - 1) {
        local_end = file_size;
    }

    FILE *fp = fopen(dataset_path, "r");
    if (!fp) {
        fprintf(stderr, "Rank %d failed to open dataset %s\n", rank, dataset_path);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (local_start > 0) {
        if (fseeko(fp, local_start, SEEK_SET) != 0) {
            fprintf(stderr, "Rank %d failed to seek to offset %lld\n", rank, local_start);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (local_start > header_len) {
            char *discard = NULL;
            size_t discard_len = 0;
            getline(&discard, &discard_len, fp);
            free(discard);
        }
    } else {
        if (fseeko(fp, header_len, SEEK_SET) != 0) {
            fprintf(stderr, "Rank %d failed to seek to header end\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    HashTable word_counts;
    HashTable artist_counts;
    ht_init(&word_counts, 65536);
    ht_init(&artist_counts, 8192);

    CountType local_word_total = 0;
    CountType local_song_total = 0;

    char *line = NULL;
    size_t line_buf = 0;

    while (1) {
        long long position = ftello(fp);
        if (position < 0) {
            break;
        }
        if (rank != world_size - 1 && position >= local_end) {
            break;
        }
        ssize_t read = getline(&line, &line_buf, fp);
        if (read < 0) {
            break;
        }
        char *artist = NULL;
        char *lyrics = NULL;
        if (!parse_csv_line(line, &artist, &lyrics)) {
            free(artist);
            free(lyrics);
            continue;
        }
        if (artist && *artist) {
            ht_put(&artist_counts, artist, 1);
        }
        if (lyrics && *lyrics) {
            process_lyrics(&word_counts, lyrics, &local_word_total);
        }
        local_song_total++;
        free(artist);
        free(lyrics);
    }

    free(line);
    fclose(fp);

    double compute_time = MPI_Wtime() - start_time;

    CountType global_word_total = 0;
    CountType global_song_total = 0;
    MPI_Reduce(&local_word_total, &global_word_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_song_total, &global_song_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ensure_output_dir(output_dir);
    }

    if (rank == 0) {
        HashTable global_words;
        HashTable global_artists;
        ht_init(&global_words, word_counts.capacity);
        ht_init(&global_artists, artist_counts.capacity);
        ht_merge(&global_words, &word_counts);
        ht_merge(&global_artists, &artist_counts);

        ht_free(&word_counts);
        ht_free(&artist_counts);

        for (int source = 1; source < world_size; ++source) {
            receive_hash_table(&global_words, source, 100, MPI_COMM_WORLD);
            receive_hash_table(&global_artists, source, 200, MPI_COMM_WORLD);
        }

        snprintf(word_output_path, sizeof(word_output_path), "%s/word_counts.csv", output_dir);
        snprintf(artist_output_path, sizeof(artist_output_path), "%s/top_artists.csv", output_dir);
        snprintf(metrics_output_path, sizeof(metrics_output_path), "%s/performance_metrics.json", output_dir);

        write_table_csv(&global_words, word_output_path, "word", word_limit);
        write_table_csv(&global_artists, artist_output_path, "artist", artist_limit);

        size_t word_array_size = 0;
        Entry *word_entries = ht_to_array(&global_words, &word_array_size);
        qsort(word_entries, word_array_size, sizeof(Entry), entry_compare_desc);
        size_t artist_array_size = 0;
        Entry *artist_entries = ht_to_array(&global_artists, &artist_array_size);
        qsort(artist_entries, artist_array_size, sizeof(Entry), entry_compare_desc);

        printf("=== Parallel Spotify Analysis ===\n");
        printf("Total songs processed: %lld\n", (long long)global_song_total);
        printf("Total words counted: %lld\n", (long long)global_word_total);
        size_t preview_words = word_array_size < 10 ? word_array_size : 10;
        printf("Top %zu words:\n", preview_words);
        for (size_t i = 0; i < preview_words; ++i) {
            printf("  %s: %lld\n", word_entries[i].key, word_entries[i].value);
        }
        size_t preview_artists = artist_array_size < 10 ? artist_array_size : 10;
        printf("Top %zu artists:\n", preview_artists);
        for (size_t i = 0; i < preview_artists; ++i) {
            printf("  %s: %lld songs\n", artist_entries[i].key, artist_entries[i].value);
        }

        free(word_entries);
        free(artist_entries);

        ht_free(&global_words);
        ht_free(&global_artists);
    } else {
        send_hash_table(&word_counts, 0, 100, MPI_COMM_WORLD);
        send_hash_table(&artist_counts, 0, 200, MPI_COMM_WORLD);
        ht_free(&word_counts);
        ht_free(&artist_counts);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - start_time;

    double sum_compute = 0.0;
    double max_compute = 0.0;
    double min_compute = 0.0;
    double sum_total = 0.0;
    double max_total = 0.0;
    double min_total = 0.0;

    MPI_Reduce(&compute_time, &sum_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &min_compute, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &min_total, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_compute = sum_compute / world_size;
        double avg_total = sum_total / world_size;
        const char *metrics_path = metrics_output_path[0] ? metrics_output_path : "output/performance_metrics.json";
        FILE *metrics_fp = fopen(metrics_path, "w");
        if (metrics_fp) {
            fprintf(metrics_fp, "{\n");
            fprintf(metrics_fp, "  \"processes\": %d,\n", world_size);
            fprintf(metrics_fp, "  \"total_songs\": %lld,\n", (long long)global_song_total);
            fprintf(metrics_fp, "  \"total_words\": %lld,\n", (long long)global_word_total);
            fprintf(metrics_fp, "  \"compute_time\": {\n");
            fprintf(metrics_fp, "    \"avg_seconds\": %.6f,\n", avg_compute);
            fprintf(metrics_fp, "    \"min_seconds\": %.6f,\n", min_compute);
            fprintf(metrics_fp, "    \"max_seconds\": %.6f\n", max_compute);
            fprintf(metrics_fp, "  },\n");
            fprintf(metrics_fp, "  \"total_time\": {\n");
            fprintf(metrics_fp, "    \"avg_seconds\": %.6f,\n", avg_total);
            fprintf(metrics_fp, "    \"min_seconds\": %.6f,\n", min_total);
            fprintf(metrics_fp, "    \"max_seconds\": %.6f\n", max_total);
            fprintf(metrics_fp, "  }\n");
            fprintf(metrics_fp, "}\n");
            fclose(metrics_fp);
        } else {
            fprintf(stderr, "Failed to write performance metrics file\n");
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
