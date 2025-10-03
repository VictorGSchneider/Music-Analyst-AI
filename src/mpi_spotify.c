#include <mpi.h>
#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define WORD_BUCKET_COUNT 131071
#define ARTIST_BUCKET_COUNT 32749
#define TAG_DATA 1
#define TAG_DONE 2

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static void fatal_error(int rank, const char *message) {
    if (rank == 0) {
        fprintf(stderr, "Error: %s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static unsigned long hash_string(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = (unsigned char)*str++) != '\0') {
        hash = ((hash << 5) + hash) + (unsigned long)c;
    }
    return hash;
}

typedef struct MapEntry {
    char *key;
    long count;
    struct MapEntry *next;
} MapEntry;

typedef struct {
    MapEntry **buckets;
    size_t bucket_count;
    size_t entry_count;
} HashMap;

static void map_init(HashMap *map, size_t bucket_count) {
    map->buckets = (MapEntry **)calloc(bucket_count, sizeof(MapEntry *));
    map->bucket_count = bucket_count;
    map->entry_count = 0;
}

static char *str_duplicate(const char *src) {
    size_t len = strlen(src);
    char *copy = (char *)malloc(len + 1);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, src, len + 1);
    return copy;
}

static void map_increment(HashMap *map, const char *key, long delta) {
    unsigned long hash_value = hash_string(key);
    size_t index = hash_value % map->bucket_count;
    MapEntry *entry = map->buckets[index];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->count += delta;
            return;
        }
        entry = entry->next;
    }

    MapEntry *new_entry = (MapEntry *)malloc(sizeof(MapEntry));
    if (!new_entry) {
        return;
    }
    new_entry->key = str_duplicate(key);
    if (!new_entry->key) {
        free(new_entry);
        return;
    }
    new_entry->count = delta;
    new_entry->next = map->buckets[index];
    map->buckets[index] = new_entry;
    map->entry_count++;
}

typedef void (*map_iter_callback)(const char *, long, void *);

static void map_foreach(HashMap *map, map_iter_callback callback, void *user_data) {
    for (size_t i = 0; i < map->bucket_count; ++i) {
        MapEntry *entry = map->buckets[i];
        while (entry) {
            callback(entry->key, entry->count, user_data);
            entry = entry->next;
        }
    }
}

static void map_free(HashMap *map) {
    if (!map->buckets) {
        return;
    }
    for (size_t i = 0; i < map->bucket_count; ++i) {
        MapEntry *entry = map->buckets[i];
        while (entry) {
            MapEntry *next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    map->buckets = NULL;
    map->bucket_count = 0;
    map->entry_count = 0;
}

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} StringBuilder;

static void sb_init(StringBuilder *sb, size_t initial_capacity) {
    sb->capacity = initial_capacity > 0 ? initial_capacity : 64;
    sb->length = 0;
    sb->data = (char *)malloc(sb->capacity);
    if (sb->data) {
        sb->data[0] = '\0';
    }
}

static void sb_reserve(StringBuilder *sb, size_t additional) {
    if (!sb->data) {
        return;
    }
    if (sb->length + additional + 1 <= sb->capacity) {
        return;
    }
    size_t new_capacity = sb->capacity * 2;
    while (new_capacity < sb->length + additional + 1) {
        new_capacity *= 2;
    }
    char *new_data = (char *)realloc(sb->data, new_capacity);
    if (!new_data) {
        return;
    }
    sb->data = new_data;
    sb->capacity = new_capacity;
}

static void sb_append_char(StringBuilder *sb, char c) {
    if (!sb->data) {
        return;
    }
    sb_reserve(sb, 1);
    sb->data[sb->length++] = c;
    sb->data[sb->length] = '\0';
}

static void sb_append_str(StringBuilder *sb, const char *str) {
    if (!sb->data || !str) {
        return;
    }
    size_t len = strlen(str);
    sb_reserve(sb, len);
    memcpy(sb->data + sb->length, str, len + 1);
    sb->length += len;
}

static char *sb_build(StringBuilder *sb) {
    if (!sb->data) {
        return NULL;
    }
    char *result = sb->data;
    sb->data = NULL;
    sb->capacity = 0;
    sb->length = 0;
    return result;
}

static void sb_reset(StringBuilder *sb) {
    sb->length = 0;
    if (sb->data) {
        sb->data[0] = '\0';
    }
}

static void sb_free(StringBuilder *sb) {
    if (sb->data) {
        free(sb->data);
        sb->data = NULL;
    }
    sb->length = 0;
    sb->capacity = 0;
}

static void trim_whitespace(char *value) {
    if (!value) {
        return;
    }
    size_t len = strlen(value);
    size_t start = 0;
    while (start < len && isspace((unsigned char)value[start])) {
        start++;
    }
    if (start == len) {
        value[0] = '\0';
        return;
    }
    size_t end = len - 1;
    while (end > start && isspace((unsigned char)value[end])) {
        end--;
    }
    size_t new_len = end - start + 1;
    if (start > 0) {
        memmove(value, value + start, new_len);
    }
    value[new_len] = '\0';
}

static int read_csv_record(FILE *fp, char ***fields_out, size_t *field_count_out) {
    if (!fp) {
        return 0;
    }
    size_t fields_capacity = 8;
    size_t field_count = 0;
    char **fields = (char **)malloc(fields_capacity * sizeof(char *));
    if (!fields) {
        return 0;
    }

    StringBuilder sb;
    sb_init(&sb, 256);

    int c;
    bool inside_quotes = false;
    bool have_data = false;

    while ((c = fgetc(fp)) != EOF) {
        have_data = true;
        if (inside_quotes) {
            if (c == '"') {
                int next = fgetc(fp);
                if (next == '"') {
                    sb_append_char(&sb, '"');
                } else {
                    inside_quotes = false;
                    if (next != EOF) {
                        if (next == '\r') {
                            int next2 = fgetc(fp);
                            if (next2 != EOF) {
                                ungetc(next2, fp);
                            }
                            next = '\n';
                        }
                        ungetc(next, fp);
                    }
                }
            } else {
                sb_append_char(&sb, (char)c);
            }
        } else {
            if (c == '"') {
                inside_quotes = true;
            } else if (c == ',') {
                if (field_count == fields_capacity) {
                    fields_capacity *= 2;
                    char **tmp = (char **)realloc(fields, fields_capacity * sizeof(char *));
                    if (!tmp) {
                        sb_free(&sb);
                        for (size_t i = 0; i < field_count; ++i) {
                            free(fields[i]);
                        }
                        free(fields);
                        return 0;
                    }
                    fields = tmp;
                }
                char *field_value = str_duplicate(sb.data ? sb.data : "");
                if (!field_value) {
                    sb_free(&sb);
                    for (size_t i = 0; i < field_count; ++i) {
                        free(fields[i]);
                    }
                    free(fields);
                    return 0;
                }
                trim_whitespace(field_value);
                fields[field_count++] = field_value;
                sb_reset(&sb);
            } else if (c == '\n') {
                if (field_count == fields_capacity) {
                    fields_capacity *= 2;
                    char **tmp = (char **)realloc(fields, fields_capacity * sizeof(char *));
                    if (!tmp) {
                        sb_free(&sb);
                        for (size_t i = 0; i < field_count; ++i) {
                            free(fields[i]);
                        }
                        free(fields);
                        return 0;
                    }
                    fields = tmp;
                }
                char *field_value = str_duplicate(sb.data ? sb.data : "");
                if (!field_value) {
                    sb_free(&sb);
                    for (size_t i = 0; i < field_count; ++i) {
                        free(fields[i]);
                    }
                    free(fields);
                    return 0;
                }
                trim_whitespace(field_value);
                fields[field_count++] = field_value;
                sb_reset(&sb);
                break;
            } else if (c == '\r') {
                continue;
            } else {
                sb_append_char(&sb, (char)c);
            }
        }
    }

    if (!have_data && sb.length == 0 && field_count == 0) {
        sb_free(&sb);
        free(fields);
        return 0;
    }

    if (sb.length > 0 || inside_quotes || field_count > 0) {
        if (field_count == fields_capacity) {
            fields_capacity *= 2;
            char **tmp = (char **)realloc(fields, fields_capacity * sizeof(char *));
            if (!tmp) {
                sb_free(&sb);
                for (size_t i = 0; i < field_count; ++i) {
                    free(fields[i]);
                }
                free(fields);
                return 0;
            }
            fields = tmp;
        }
        char *field_value = str_duplicate(sb.data ? sb.data : "");
        if (!field_value) {
            sb_free(&sb);
            for (size_t i = 0; i < field_count; ++i) {
                free(fields[i]);
            }
            free(fields);
            return 0;
        }
        trim_whitespace(field_value);
        fields[field_count++] = field_value;
    }

    sb_free(&sb);
    *fields_out = fields;
    *field_count_out = field_count;
    return 1;
}

static void free_csv_fields(char **fields, size_t field_count) {
    if (!fields) {
        return;
    }
    for (size_t i = 0; i < field_count; ++i) {
        free(fields[i]);
    }
    free(fields);
}

static void tokenize_text(HashMap *word_counts, const char *text) {
    if (!text) {
        return;
    }
    size_t len = strlen(text);
    if (len == 0) {
        return;
    }
    char *buffer = (char *)malloc(len + 1);
    if (!buffer) {
        return;
    }
    size_t index = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (isalnum(c)) {
            buffer[index++] = (char)tolower(c);
        } else {
            if (index > 0) {
                buffer[index] = '\0';
                map_increment(word_counts, buffer, 1);
                index = 0;
            }
        }
    }
    if (index > 0) {
        buffer[index] = '\0';
        map_increment(word_counts, buffer, 1);
    }
    free(buffer);
}

static void process_record(HashMap *word_counts, HashMap *artist_counts, const char *artist, const char *text) {
    if (artist && *artist) {
        map_increment(artist_counts, artist, 1);
    }
    tokenize_text(word_counts, text);
}

static void ensure_directory_exists(int rank, const char *path) {
    if (rank != 0 || !path) {
        return;
    }
    struct stat st;
    if (stat(path, &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            fprintf(stderr, "Error: %s exists and is not a directory\n", path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return;
    }
    if (mkdir(path, 0755) != 0) {
        if (errno != EEXIST) {
            fprintf(stderr, "Error creating directory %s: %s\n", path, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

typedef struct {
    const char *input_path;
    const char *output_dir;
    long max_records;
    long top_word_limit;
    long top_artist_limit;
} ProgramOptions;

static void options_set_defaults(ProgramOptions *options) {
    options->input_path = NULL;
    options->output_dir = "output";
    options->max_records = -1;
    options->top_word_limit = 0;
    options->top_artist_limit = 0;
}

static void print_usage(void) {
    fprintf(stderr,
            "Usage: mpirun -np <processes> ./mpi_spotify --input <csv_path> [options]\n"
            "Options:\n"
            "  --output-dir <path>       Directory to store output files (default: output)\n"
            "  --max-records <n>         Limit the number of records processed (useful for testing)\n"
            "  --top-words <n>           Limit of most frequent words to save (0 = all)\n"
            "  --top-artists <n>         Limit of artists to save (0 = all)\n");
}

static int parse_long(const char *value, long *out) {
    char *endptr = NULL;
    long parsed = strtol(value, &endptr, 10);
    if (!endptr || *endptr != '\0') {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int parse_arguments(int argc, char **argv, ProgramOptions *options, int rank) {
    options_set_defaults(options);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                if (rank == 0) {
                    fprintf(stderr, "--input requires a value\n");
                }
                return 0;
            }
            options->input_path = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0) {
            if (i + 1 >= argc) {
                if (rank == 0) {
                    fprintf(stderr, "--output-dir requires a value\n");
                }
                return 0;
            }
            options->output_dir = argv[++i];
        } else if (strcmp(argv[i], "--max-records") == 0) {
            if (i + 1 >= argc) {
                if (rank == 0) {
                    fprintf(stderr, "--max-records requires a value\n");
                }
                return 0;
            }
            if (!parse_long(argv[i + 1], &options->max_records)) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid value for --max-records: %s\n", argv[i + 1]);
                }
                return 0;
            }
            ++i;
        } else if (strcmp(argv[i], "--top-words") == 0) {
            if (i + 1 >= argc) {
                if (rank == 0) {
                    fprintf(stderr, "--top-words requires a value\n");
                }
                return 0;
            }
            if (!parse_long(argv[i + 1], &options->top_word_limit)) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid value for --top-words: %s\n", argv[i + 1]);
                }
                return 0;
            }
            ++i;
        } else if (strcmp(argv[i], "--top-artists") == 0) {
            if (i + 1 >= argc) {
                if (rank == 0) {
                    fprintf(stderr, "--top-artists requires a value\n");
                }
                return 0;
            }
            if (!parse_long(argv[i + 1], &options->top_artist_limit)) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid value for --top-artists: %s\n", argv[i + 1]);
                }
                return 0;
            }
            ++i;
        } else {
            if (rank == 0) {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            }
            return 0;
        }
    }
    if (!options->input_path) {
        if (rank == 0) {
            fprintf(stderr, "Missing --input argument\n");
        }
        return 0;
    }
    return 1;
}

typedef struct {
    char *data;
    size_t size;
} SerializedMap;

typedef struct {
    StringBuilder *builder;
} SerializeContext;

static void serialize_callback(const char *key, long value, void *user_data) {
    SerializeContext *ctx = (SerializeContext *)user_data;
    char count_buffer[64];
    snprintf(count_buffer, sizeof(count_buffer), "%ld", value);
    sb_append_str(ctx->builder, key);
    sb_append_char(ctx->builder, '\t');
    sb_append_str(ctx->builder, count_buffer);
    sb_append_char(ctx->builder, '\n');
}

static void serialize_map(HashMap *map, SerializedMap *out) {
    StringBuilder sb;
    sb_init(&sb, map->entry_count * 16 + 64);
    SerializeContext ctx = {&sb};
    map_foreach(map, serialize_callback, &ctx);
    out->data = sb_build(&sb);
    out->size = out->data ? strlen(out->data) : 0;
}

static void deserialize_and_merge(HashMap *map, const char *data, size_t size) {
    if (!data || size == 0) {
        return;
    }
    size_t start = 0;
    while (start < size) {
        size_t end = start;
        while (end < size && data[end] != '\n') {
            end++;
        }
        if (end == start) {
            start = end + 1;
            continue;
        }
        size_t line_length = end - start;
        char *line = (char *)malloc(line_length + 1);
        if (!line) {
            break;
        }
        memcpy(line, data + start, line_length);
        line[line_length] = '\0';
        char *sep = strchr(line, '\t');
        if (sep) {
            *sep = '\0';
            const char *key = line;
            const char *value_str = sep + 1;
            long value = strtol(value_str, NULL, 10);
            map_increment(map, key, value);
        }
        free(line);
        start = end + 1;
    }
}

typedef struct {
    char *key;
    long value;
} Pair;

static int compare_pairs_desc(const void *a, const void *b) {
    const Pair *pa = (const Pair *)a;
    const Pair *pb = (const Pair *)b;
    if (pa->value < pb->value) {
        return 1;
    }
    if (pa->value > pb->value) {
        return -1;
    }
    return strcmp(pa->key, pb->key);
}

typedef struct {
    Pair *pairs;
    size_t index;
} CollectContext;

static void collect_callback(const char *key, long value, void *user_data) {
    CollectContext *ctx = (CollectContext *)user_data;
    ctx->pairs[ctx->index].key = str_duplicate(key);
    ctx->pairs[ctx->index].value = value;
    ctx->index++;
}

static Pair *collect_pairs(HashMap *map, size_t *count_out) {
    Pair *pairs = (Pair *)malloc(map->entry_count * sizeof(Pair));
    if (!pairs) {
        *count_out = 0;
        return NULL;
    }
    CollectContext ctx;
    ctx.pairs = pairs;
    ctx.index = 0;
    map_foreach(map, collect_callback, &ctx);
    *count_out = ctx.index;
    return pairs;
}

static void free_pairs(Pair *pairs, size_t count) {
    if (!pairs) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free(pairs[i].key);
    }
    free(pairs);
}

static char *csv_escape(const char *value) {
    if (!value) {
        return str_duplicate("");
    }
    bool needs_quotes = false;
    size_t len = 0;
    for (const char *p = value; *p; ++p) {
        if (*p == '"' || *p == ',' || *p == '\n' || *p == '\r') {
            needs_quotes = true;
        }
        len++;
    }
    if (!needs_quotes) {
        return str_duplicate(value);
    }
    char *escaped = (char *)malloc(len * 2 + 3);
    if (!escaped) {
        return NULL;
    }
    char *out = escaped;
    *out++ = '"';
    for (const char *p = value; *p; ++p) {
        if (*p == '"') {
            *out++ = '"';
        }
        *out++ = *p;
    }
    *out++ = '"';
    *out = '\0';
    return escaped;
}

static void write_pairs_to_csv(const char *path, const char *header_key, const char *header_value, Pair *pairs, size_t count, long limit) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing: %s\n", path, strerror(errno));
        return;
    }
    fprintf(fp, "%s,%s\n", header_key, header_value);
    size_t to_write = count;
    if (limit > 0 && (long)count > limit) {
        to_write = (size_t)limit;
    }
    for (size_t i = 0; i < to_write; ++i) {
        char *escaped_key = csv_escape(pairs[i].key);
        if (!escaped_key) {
            continue;
        }
        fprintf(fp, "%s,%ld\n", escaped_key, pairs[i].value);
        free(escaped_key);
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ProgramOptions options;
    if (!parse_arguments(argc, argv, &options, world_rank)) {
        if (world_rank == 0) {
            print_usage();
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ensure_directory_exists(world_rank, options.output_dir);

    HashMap word_counts;
    HashMap artist_counts;
    map_init(&word_counts, WORD_BUCKET_COUNT);
    map_init(&artist_counts, ARTIST_BUCKET_COUNT);

    long local_records = 0;
    double processing_time = 0.0;
    double total_start = MPI_Wtime();

    if (world_rank == 0) {
        FILE *fp = fopen(options.input_path, "r");
        if (!fp) {
            fprintf(stderr, "Failed to open %s: %s\n", options.input_path, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        char **header_fields = NULL;
        size_t header_count = 0;
        if (!read_csv_record(fp, &header_fields, &header_count)) {
            fprintf(stderr, "Failed to read CSV header\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        free_csv_fields(header_fields, header_count);

        long records_processed = 0;
        int next_worker = 1;
        while (1) {
            if (options.max_records >= 0 && records_processed >= options.max_records) {
                break;
            }
            char **fields = NULL;
            size_t field_count = 0;
            if (!read_csv_record(fp, &fields, &field_count)) {
                break;
            }
            if (field_count < 4) {
                free_csv_fields(fields, field_count);
                continue;
            }
            const char *artist = fields[0];
            const char *text = fields[3];

            int target_rank = (world_size == 1) ? 0 : next_worker;
            if (target_rank == 0) {
                double start = MPI_Wtime();
                process_record(&word_counts, &artist_counts, artist, text);
                processing_time += MPI_Wtime() - start;
                local_records++;
            } else {
                int lengths[2];
                lengths[0] = (int)strlen(artist);
                lengths[1] = (int)strlen(text);
                MPI_Send(lengths, 2, MPI_INT, target_rank, TAG_DATA, MPI_COMM_WORLD);
                MPI_Send(artist, lengths[0], MPI_CHAR, target_rank, TAG_DATA, MPI_COMM_WORLD);
                MPI_Send(text, lengths[1], MPI_CHAR, target_rank, TAG_DATA, MPI_COMM_WORLD);
                next_worker++;
                if (next_worker >= world_size) {
                    next_worker = 1;
                }
            }
            records_processed++;
            free_csv_fields(fields, field_count);
        }
        fclose(fp);

        if (world_size > 1) {
            for (int worker = 1; worker < world_size; ++worker) {
                MPI_Send(NULL, 0, MPI_INT, worker, TAG_DONE, MPI_COMM_WORLD);
            }
        }
    } else {
        while (1) {
            MPI_Status status;
            int lengths[2];
            MPI_Recv(lengths, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_DONE) {
                break;
            }
            int artist_len = lengths[0];
            int text_len = lengths[1];
            char *artist = (char *)malloc((size_t)artist_len + 1);
            char *text = (char *)malloc((size_t)text_len + 1);
            if (!artist || !text) {
                fatal_error(world_rank, "Memory allocation failure");
            }
            MPI_Recv(artist, artist_len, MPI_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(text, text_len, MPI_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            artist[artist_len] = '\0';
            text[text_len] = '\0';
            double start = MPI_Wtime();
            process_record(&word_counts, &artist_counts, artist, text);
            processing_time += MPI_Wtime() - start;
            local_records++;
            free(artist);
            free(text);
        }
    }

    double total_time = MPI_Wtime() - total_start;

    SerializedMap serialized_words = {NULL, 0};
    SerializedMap serialized_artists = {NULL, 0};
    serialize_map(&word_counts, &serialized_words);
    serialize_map(&artist_counts, &serialized_artists);

    int word_length = (int)serialized_words.size;
    int artist_length = (int)serialized_artists.size;

    int *word_lengths = NULL;
    int *artist_lengths = NULL;
    double *processing_times = NULL;
    double *total_times = NULL;
    long *records_counts = NULL;

    if (world_rank == 0) {
        word_lengths = (int *)malloc(world_size * sizeof(int));
        artist_lengths = (int *)malloc(world_size * sizeof(int));
        processing_times = (double *)malloc(world_size * sizeof(double));
        total_times = (double *)malloc(world_size * sizeof(double));
        records_counts = (long *)malloc(world_size * sizeof(long));
    }

    MPI_Gather(&word_length, 1, MPI_INT, word_lengths, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&artist_length, 1, MPI_INT, artist_lengths, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&processing_time, 1, MPI_DOUBLE, processing_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&total_time, 1, MPI_DOUBLE, total_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_records, 1, MPI_LONG, records_counts, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    char *word_recv_buffer = NULL;
    char *artist_recv_buffer = NULL;
    int *word_displs = NULL;
    int *artist_displs = NULL;

    if (world_rank == 0) {
        int total_word_bytes = 0;
        int total_artist_bytes = 0;
        word_displs = (int *)malloc(world_size * sizeof(int));
        artist_displs = (int *)malloc(world_size * sizeof(int));
        for (int i = 0; i < world_size; ++i) {
            word_displs[i] = total_word_bytes;
            total_word_bytes += word_lengths[i];
            artist_displs[i] = total_artist_bytes;
            total_artist_bytes += artist_lengths[i];
        }
        word_recv_buffer = (char *)malloc((size_t)total_word_bytes);
        artist_recv_buffer = (char *)malloc((size_t)total_artist_bytes);
    }

    MPI_Gatherv(serialized_words.data, word_length, MPI_CHAR,
                word_recv_buffer, word_lengths, word_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(serialized_artists.data, artist_length, MPI_CHAR,
                artist_recv_buffer, artist_lengths, artist_displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        HashMap global_word_counts;
        HashMap global_artist_counts;
        map_init(&global_word_counts, WORD_BUCKET_COUNT * 2);
        map_init(&global_artist_counts, ARTIST_BUCKET_COUNT * 2);
        for (int i = 0; i < world_size; ++i) {
            const char *word_data = word_recv_buffer + word_displs[i];
            int word_size = word_lengths[i];
            deserialize_and_merge(&global_word_counts, word_data, (size_t)word_size);
            const char *artist_data = artist_recv_buffer + artist_displs[i];
            int artist_size = artist_lengths[i];
            deserialize_and_merge(&global_artist_counts, artist_data, (size_t)artist_size);
        }

        size_t word_pair_count = 0;
        Pair *word_pairs = collect_pairs(&global_word_counts, &word_pair_count);
        qsort(word_pairs, word_pair_count, sizeof(Pair), compare_pairs_desc);

        size_t artist_pair_count = 0;
        Pair *artist_pairs = collect_pairs(&global_artist_counts, &artist_pair_count);
        qsort(artist_pairs, artist_pair_count, sizeof(Pair), compare_pairs_desc);

        char word_output_path[PATH_MAX];
        snprintf(word_output_path, sizeof(word_output_path), "%s/word_counts.csv", options.output_dir);
        write_pairs_to_csv(word_output_path, "word", "count", word_pairs, word_pair_count, options.top_word_limit);

        char artist_output_path[PATH_MAX];
        snprintf(artist_output_path, sizeof(artist_output_path), "%s/artist_song_counts.csv", options.output_dir);
        write_pairs_to_csv(artist_output_path, "artist", "song_count", artist_pairs, artist_pair_count, options.top_artist_limit);

        printf("=== Runtime Metrics ===\n");
        printf("Rank, Records, ProcessingTime(s), TotalTime(s)\n");
        for (int i = 0; i < world_size; ++i) {
            printf("%d,%ld,%.6f,%.6f\n", i, records_counts[i], processing_times[i], total_times[i]);
        }

        map_free(&global_word_counts);
        map_free(&global_artist_counts);
        free_pairs(word_pairs, word_pair_count);
        free_pairs(artist_pairs, artist_pair_count);
    }

    free(serialized_words.data);
    free(serialized_artists.data);
    map_free(&word_counts);
    map_free(&artist_counts);
    if (world_rank == 0) {
        free(word_lengths);
        free(artist_lengths);
        free(processing_times);
        free(total_times);
        free(records_counts);
        free(word_recv_buffer);
        free(artist_recv_buffer);
        free(word_displs);
        free(artist_displs);
    }

    MPI_Finalize();
    return 0;
}
