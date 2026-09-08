import re
import tarfile

re_clean_path = re.compile(r"(?:^\./|/\.(?=/))")


def retrieve_media_source(media_path, media_sources, aux_data_prefixes):
    """retrieve the appropriate media url so that media file can be
    access with os.path.join(media_url, media_path)
    """
    path = re_clean_path.sub("", media_path)
    for prefix, aux_key in aux_data_prefixes.items():
        if path.startswith(prefix):
            if aux_key in media_sources:
                return media_sources[aux_key]
    return None
