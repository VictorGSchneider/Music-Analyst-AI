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
#include <limits.h>

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
 * Duplica um campo CSV removendo espaços excedentes e, opcionalmente,
 * preservando as aspas externas exatamente como no arquivo original. O
 * chamador é responsável por liberar o ponteiro retornado.
 */
static char *duplicate_field(const char *field, int preserve_outer_quotes) {
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
    char *result = (char *)malloc(end - start + 1);
    if (!result) {
        fprintf(stderr, "Failed to allocate memory for CSV field\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    size_t j = 0;
    if (preserve_outer_quotes && quoted) {
        for (size_t i = start; i < end; ++i) {
            result[j++] = field[i];
        }
    } else {
        size_t inner_start = start;
        size_t inner_end = end;
        if (quoted) {
            inner_start++;
            inner_end--;
        }
        for (size_t i = inner_start; i < inner_end; ++i) {
            if (field[i] == '"' && i + 1 < inner_end && field[i + 1] == '"') {
                result[j++] = '"';
                i++;
            } else {
                result[j++] = field[i];
            }
        }
    }
    result[j] = '\0';
    trim_inplace(result);
    return result;
}

/* Extrai artista e letra a partir de uma linha CSV, respeitando as aspas originais. */
static int parse_csv_line(const char *line, char **artist_out, char **lyrics_out,
                          int preserve_artist_quotes, int preserve_lyrics_quotes) {
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
    *artist_out = duplicate_field(fields[0], preserve_artist_quotes);
    *lyrics_out = duplicate_field(fields[3], preserve_lyrics_quotes);
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

/*
 * Tokeniza as letras, acumula contagem por palavra e atualiza o total geral,
 * preservando apóstrofos para não descaracterizar contrações e variações.
 */
static void process_lyrics(HashTable *word_counts, const char *lyrics, CountType *total_words) {
    size_t capacity = 64;
    char *buffer = (char *)malloc(capacity);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer for tokenisation\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    size_t length = 0;
    for (const unsigned char *p = (const unsigned char *)lyrics; *p; ++p) {
        if (isalnum(*p) || *p == '\'') {
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
            if (isalnum(*p)) {
                buffer[length++] = (char)tolower(*p);
            } else {
                buffer[length++] = (char)*p;
            }
        } else {
            if (length > 0) {
                buffer[length] = '\0';
                if (length >= 3) {
                    ht_put(word_counts, buffer, 1);
                    (*total_words)++;
                }
                length = 0;
            }
        }
    }
    if (length > 0) {
        buffer[length] = '\0';
        if (length >= 3) {
            ht_put(word_counts, buffer, 1);
            (*total_words)++;
        }
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

/*
 * Cria diretórios recursivamente, garantindo que todos os níveis do caminho
 * existam antes do uso. Retorna 0 em sucesso e -1 em caso de erro.
 */
static int ensure_directory_recursive(const char *path) {
    if (!path || !*path) {
        return 0;
    }
    size_t len = strlen(path);
    if (len >= PATH_MAX) {
        errno = ENAMETOOLONG;
        return -1;
    }
    char buffer[PATH_MAX];
    memcpy(buffer, path, len + 1);
    for (size_t i = 1; i < len; ++i) {
        if (buffer[i] == '/' || buffer[i] == '\\') {
            char saved = buffer[i];
            buffer[i] = '\0';
            if (buffer[0] != '\0' && strcmp(buffer, ".") != 0) {
                if (MKDIR(buffer) != 0 && errno != EEXIST) {
                    buffer[i] = saved;
                    return -1;
                }
            }
            buffer[i] = saved;
        }
    }
    if (MKDIR(buffer) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

/*
 * Normaliza o nome do cabeçalho para servir como base de nome de arquivo,
 * substituindo caracteres problemáticos por sublinhados.
 */
static void sanitize_header_name(const char *input, char *output, size_t output_len) {
    if (!output || output_len == 0) {
        return;
    }
    size_t j = 0;
    if (input) {
        for (size_t i = 0; input[i] != '\0'; ++i) {
            unsigned char c = (unsigned char)input[i];
            if (c == '\n' || c == '\r') {
                continue;
            }
            if (isspace(c)) {
                if (j + 1 < output_len) {
                    output[j++] = '_';
                }
            } else if (isalnum(c) || c == '-' || c == '.' || c == '_') {
                if (j + 1 < output_len) {
                    output[j++] = (char)c;
                }
            } else {
                if (j + 1 < output_len) {
                    output[j++] = '_';
                }
            }
        }
    }
    if (j == 0) {
        const char *fallback = "col";
        for (size_t i = 0; fallback[i] != '\0' && j + 1 < output_len; ++i) {
            output[j++] = fallback[i];
        }
    }
    output[j] = '\0';
}

/*
 * Lê um registro completo do CSV, respeitando quebras de linha dentro de
 * campos entre aspas. Retorna o total de bytes lidos ou -1 no fim do arquivo.
 */
static ssize_t read_csv_record(FILE *fp, char **buffer, size_t *buf_size) {
    if (!fp) {
        return -1;
    }
    if (!buffer || !buf_size) {
        return -1;
    }
    if (!*buffer || *buf_size == 0) {
        *buf_size = 1024;
        *buffer = (char *)malloc(*buf_size);
        if (!*buffer) {
            fprintf(stderr, "Failed to allocate CSV buffer\n");
            return -1;
        }
    }
    size_t pos = 0;
    int in_quotes = 0;
    while (1) {
        int ch = fgetc(fp);
        if (ch == EOF) {
            if (pos == 0) {
                return -1;
            }
            break;
        }
        if (pos + 2 >= *buf_size) {
            size_t new_size = *buf_size * 2U;
            char *tmp = (char *)realloc(*buffer, new_size);
            if (!tmp) {
                fprintf(stderr, "Failed to grow CSV buffer\n");
                return -1;
            }
            *buffer = tmp;
            *buf_size = new_size;
        }
        (*buffer)[pos++] = (char)ch;
        if (ch == '"') {
            if (!in_quotes) {
                in_quotes = 1;
            } else {
                int next = fgetc(fp);
                if (next == '"') {
                    if (pos + 1 >= *buf_size) {
                        size_t new_size = *buf_size * 2U;
                        char *tmp = (char *)realloc(*buffer, new_size);
                        if (!tmp) {
                            fprintf(stderr, "Failed to grow CSV buffer\n");
                            return -1;
                        }
                        *buffer = tmp;
                        *buf_size = new_size;
                    }
                    (*buffer)[pos++] = '"';
                } else {
                    if (next != EOF) {
                        ungetc(next, fp);
                    }
                    in_quotes = 0;
                }
            }
        } else if ((ch == '\n' || ch == '\r') && !in_quotes) {
            if (ch == '\r') {
                int next = fgetc(fp);
                if (next == '\n') {
                    if (pos + 1 >= *buf_size) {
                        size_t new_size = *buf_size * 2U;
                        char *tmp = (char *)realloc(*buffer, new_size);
                        if (!tmp) {
                            fprintf(stderr, "Failed to grow CSV buffer\n");
                            return -1;
                        }
                        *buffer = tmp;
                        *buf_size = new_size;
                    }
                    (*buffer)[pos++] = (char)next;
                } else if (next != EOF) {
                    ungetc(next, fp);
                }
            }
            break;
        }
    }
    (*buffer)[pos] = '\0';
    return (ssize_t)pos;
}

/*
 * Cria arquivos separados para as colunas de artistas e letras, mantendo as
 * aspas originais dos campos de texto. Retorna 1 em sucesso e 0 em caso de
 * falha.
 */
static int split_dataset_columns(const char *dataset_path, const char *split_dir,
                                 const char *artist_base_name, const char *text_base_name,
                                 const char *artist_header_label, const char *text_header_label,
                                 char *artist_out_path, size_t artist_out_len,
                                 char *text_out_path, size_t text_out_len) {
    if (!dataset_path || !split_dir || !artist_base_name || !text_base_name) {
        return 0;
    }
    if (!artist_out_path || !text_out_path) {
        return 0;
    }
    if (ensure_directory_recursive(split_dir) != 0) {
        fprintf(stderr, "Failed to create split directory %s: %s\n", split_dir, strerror(errno));
        return 0;
    }
    int needed = snprintf(artist_out_path, artist_out_len, "%s/%s.csv", split_dir, artist_base_name);
    if (needed < 0 || (size_t)needed >= artist_out_len) {
        fprintf(stderr, "Artist split path is too long\n");
        return 0;
    }
    needed = snprintf(text_out_path, text_out_len, "%s/%s.csv", split_dir, text_base_name);
    if (needed < 0 || (size_t)needed >= text_out_len) {
        fprintf(stderr, "Text split path is too long\n");
        return 0;
    }

    FILE *input = fopen(dataset_path, "r");
    if (!input) {
        fprintf(stderr, "Failed to open dataset %s for splitting\n", dataset_path);
        return 0;
    }
    FILE *artist_fp = fopen(artist_out_path, "w");
    FILE *text_fp = fopen(text_out_path, "w");
    if (!artist_fp || !text_fp) {
        fprintf(stderr, "Failed to create split files in %s\n", split_dir);
        if (artist_fp) {
            fclose(artist_fp);
        }
        if (text_fp) {
            fclose(text_fp);
        }
        fclose(input);
        return 0;
    }

    fprintf(artist_fp, "%s\n", (artist_header_label && *artist_header_label) ? artist_header_label : "Artists");
    fprintf(text_fp, "%s\n", (text_header_label && *text_header_label) ? text_header_label : "Texts");

    char *line = NULL;
    size_t line_cap = 0;
    ssize_t read = read_csv_record(input, &line, &line_cap); /* descarta cabeçalho */
    if (read < 0) {
        free(line);
        fclose(artist_fp);
        fclose(text_fp);
        fclose(input);
        return 1; /* dataset vazio, apenas cabeçalhos criados */
    }

    while ((read = read_csv_record(input, &line, &line_cap)) >= 0) {
        if (read == 0) {
            continue;
        }
        char *artist_raw = NULL;
        char *lyrics_raw = NULL;
        if (!parse_csv_line(line, &artist_raw, &lyrics_raw, 1, 1)) {
            free(artist_raw);
            free(lyrics_raw);
            continue;
        }
        fprintf(artist_fp, "%s\n", artist_raw ? artist_raw : "");
        fprintf(text_fp, "%s\n", lyrics_raw ? lyrics_raw : "");
        free(artist_raw);
        free(lyrics_raw);
    }

    free(line);
    fclose(artist_fp);
    fclose(text_fp);
    fclose(input);
    return 1;
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
    char output_dir[PATH_MAX];
    snprintf(output_dir, sizeof(output_dir), "output");
    char word_output_path[PATH_MAX] = {0};
    char artist_output_path[PATH_MAX] = {0};
    char metrics_output_path[PATH_MAX] = {0};
    char split_dir[PATH_MAX] = {0};
    char sanitized_artist[128] = {0};
    char sanitized_text[128] = {0};
    char artist_header_label[128] = {0};
    char text_header_label[128] = {0};
    char artist_split_path[PATH_MAX] = {0};
    char text_split_path[PATH_MAX] = {0};

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--word-limit") == 0 && i + 1 < argc) {
            word_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--artist-limit") == 0 && i + 1 < argc) {
            artist_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            strncpy(output_dir, argv[++i], sizeof(output_dir) - 1);
            output_dir[sizeof(output_dir) - 1] = '\0';
        } else if (rank == 0) {
            fprintf(stderr, "Ignoring unknown argument: %s\n", argv[i]);
        }
    }

    int split_dir_len = snprintf(split_dir, sizeof(split_dir), "%s/split_columns", output_dir);
    if (split_dir_len < 0 || (size_t)split_dir_len >= sizeof(split_dir)) {
        if (rank == 0) {
            fprintf(stderr, "Split directory path is too long\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        if (ensure_directory_recursive(output_dir) != 0) {
            fprintf(stderr, "Failed to prepare output directory %s: %s\n", output_dir, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (ensure_directory_recursive(split_dir) != 0) {
            fprintf(stderr, "Failed to prepare split directory %s: %s\n", split_dir, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        FILE *header_fp = fopen(dataset_path, "r");
        if (!header_fp) {
            fprintf(stderr, "Failed to open dataset %s\n", dataset_path);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        char *header_line = NULL;
        size_t header_cap = 0;
        ssize_t header_read = read_csv_record(header_fp, &header_line, &header_cap);
        if (header_read < 0) {
            fprintf(stderr, "Dataset does not contain a header row\n");
            free(header_line);
            fclose(header_fp);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        char *artist_header_tmp = NULL;
        char *text_header_tmp = NULL;
        if (!parse_csv_line(header_line, &artist_header_tmp, &text_header_tmp, 0, 0)) {
            fprintf(stderr, "Unable to parse dataset header\n");
            free(header_line);
            fclose(header_fp);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        strncpy(artist_header_label, artist_header_tmp, sizeof(artist_header_label) - 1);
        strncpy(text_header_label, text_header_tmp, sizeof(text_header_label) - 1);
        artist_header_label[sizeof(artist_header_label) - 1] = '\0';
        text_header_label[sizeof(text_header_label) - 1] = '\0';
        sanitize_header_name(artist_header_tmp, sanitized_artist, sizeof(sanitized_artist));
        sanitize_header_name(text_header_tmp, sanitized_text, sizeof(sanitized_text));
        free(artist_header_tmp);
        free(text_header_tmp);
        free(header_line);
        fclose(header_fp);

        if (!split_dataset_columns(dataset_path, split_dir, sanitized_artist, sanitized_text,
                                   artist_header_label, text_header_label,
                                   artist_split_path, sizeof(artist_split_path),
                                   text_split_path, sizeof(text_split_path))) {
            fprintf(stderr, "Failed to split dataset columns\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(sanitized_artist, (int)sizeof(sanitized_artist), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(sanitized_text, (int)sizeof(sanitized_text), MPI_CHAR, 0, MPI_COMM_WORLD);

    int artist_path_len = snprintf(artist_split_path, sizeof(artist_split_path), "%s/%s.csv", split_dir, sanitized_artist);
    if (artist_path_len < 0 || (size_t)artist_path_len >= sizeof(artist_split_path)) {
        if (rank == 0) {
            fprintf(stderr, "Artist split path is too long\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int text_path_len = snprintf(text_split_path, sizeof(text_split_path), "%s/%s.csv", split_dir, sanitized_text);
    if (text_path_len < 0 || (size_t)text_path_len >= sizeof(text_split_path)) {
        if (rank == 0) {
            fprintf(stderr, "Text split path is too long\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    long long text_header_len = compute_header_length(text_split_path);
    long long text_file_size = get_file_size(text_split_path);
    if (text_header_len < 0 || text_file_size < 0) {
        fprintf(stderr, "Rank %d failed to obtain text column metadata\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    long long artist_header_len = compute_header_length(artist_split_path);
    long long artist_file_size = get_file_size(artist_split_path);
    if (artist_header_len < 0 || artist_file_size < 0) {
        fprintf(stderr, "Rank %d failed to obtain artist column metadata\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    long long text_data_bytes = text_file_size > text_header_len ? text_file_size - text_header_len : 0;
    long long text_base_chunk = text_data_bytes / world_size;
    long long text_remainder = text_data_bytes % world_size;
    long long text_local_start = text_header_len + rank * text_base_chunk + (rank < text_remainder ? rank : text_remainder);
    long long text_local_end = text_local_start + text_base_chunk + (rank < text_remainder ? 1 : 0);
    if (rank == world_size - 1) {
        text_local_end = text_file_size;
    }

    long long artist_data_bytes = artist_file_size > artist_header_len ? artist_file_size - artist_header_len : 0;
    long long artist_base_chunk = artist_data_bytes / world_size;
    long long artist_remainder = artist_data_bytes % world_size;
    long long artist_local_start = artist_header_len + rank * artist_base_chunk + (rank < artist_remainder ? rank : artist_remainder);
    long long artist_local_end = artist_local_start + artist_base_chunk + (rank < artist_remainder ? 1 : 0);
    if (rank == world_size - 1) {
        artist_local_end = artist_file_size;
    }

    HashTable word_counts;
    HashTable artist_counts;
    ht_init(&word_counts, 65536);
    ht_init(&artist_counts, 8192);

    CountType local_word_total = 0;
    CountType local_song_total = 0;

    char *line = NULL;
    size_t line_buf = 0;

    FILE *text_fp = fopen(text_split_path, "r");
    if (!text_fp) {
        fprintf(stderr, "Rank %d failed to open text column %s\n", rank, text_split_path);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (text_local_start > text_header_len) {
        if (fseeko(text_fp, text_local_start, SEEK_SET) != 0) {
            fprintf(stderr, "Rank %d failed to seek text column offset %lld\n", rank, text_local_start);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (text_local_start > text_header_len) {
            if (read_csv_record(text_fp, &line, &line_buf) < 0) {
                /* offset apontou além do fim */
            }
        }
    } else {
        if (fseeko(text_fp, text_header_len, SEEK_SET) != 0) {
            fprintf(stderr, "Rank %d failed to seek to text header end\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    while (1) {
        long long position = ftello(text_fp);
        if (position < 0) {
            break;
        }
        if (rank != world_size - 1 && position >= text_local_end) {
            break;
        }
        ssize_t read_len = read_csv_record(text_fp, &line, &line_buf);
        if (read_len < 0) {
            break;
        }
        if (read_len == 0) {
            continue;
        }
        while (read_len > 0 && (line[read_len - 1] == '\n' || line[read_len - 1] == '\r')) {
            line[--read_len] = '\0';
        }
        char *lyrics = duplicate_field(line, 1);
        if (lyrics && *lyrics) {
            process_lyrics(&word_counts, lyrics, &local_word_total);
        }
        free(lyrics);
    }
    free(line);
    fclose(text_fp);

    line = NULL;
    line_buf = 0;

    FILE *artist_fp = fopen(artist_split_path, "r");
    if (!artist_fp) {
        fprintf(stderr, "Rank %d failed to open artist column %s\n", rank, artist_split_path);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (artist_local_start > artist_header_len) {
        if (fseeko(artist_fp, artist_local_start, SEEK_SET) != 0) {
            fprintf(stderr, "Rank %d failed to seek artist column offset %lld\n", rank, artist_local_start);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (artist_local_start > artist_header_len) {
            if (read_csv_record(artist_fp, &line, &line_buf) < 0) {
                /* offset apontou além do fim */
            }
        }
    } else {
        if (fseeko(artist_fp, artist_header_len, SEEK_SET) != 0) {
            fprintf(stderr, "Rank %d failed to seek to artist header end\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    while (1) {
        long long position = ftello(artist_fp);
        if (position < 0) {
            break;
        }
        if (rank != world_size - 1 && position >= artist_local_end) {
            break;
        }
        ssize_t read_len = read_csv_record(artist_fp, &line, &line_buf);
        if (read_len < 0) {
            break;
        }
        if (read_len == 0) {
            continue;
        }
        while (read_len > 0 && (line[read_len - 1] == '\n' || line[read_len - 1] == '\r')) {
            line[--read_len] = '\0';
        }
        char *artist = duplicate_field(line, 0);
        if (artist && *artist) {
            ht_put(&artist_counts, artist, 1);
        }
        local_song_total++;
        free(artist);
    }

    free(line);
    fclose(artist_fp);

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
