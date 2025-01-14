import insightface
import pkg_resources

def main():
    try:
        # Obtenir la version de insightface
        insightface_version = pkg_resources.get_distribution("insightface").version
        print(f"Version de insightface : {insightface_version}")
    except pkg_resources.DistributionNotFound:
        print("insightface n'est pas installé.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    main()

