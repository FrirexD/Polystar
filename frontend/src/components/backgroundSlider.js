import React, { useState, useEffect } from 'react';
import ImageUpload from './imageUpload';

const BackgroundSlider = () => {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [rows, setRows] = useState(1);

  const OMDB_API_KEY = 'YOUR-API-KEY';

  const shuffleArray = (array) => {
    const newArray = [...array];
    for (let i = newArray.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
    }
    return newArray;
  };

  useEffect(() => {
    const calculateRows = () => {
      const posterHeight = 120;
      const gap = 0;
      const screenHeight = window.innerHeight;
      const possibleRows = Math.floor(screenHeight / (posterHeight + gap));
      setRows(possibleRows);
    };

    calculateRows();
    window.addEventListener('resize', calculateRows);
    return () => window.removeEventListener('resize', calculateRows);
  }, []);

  useEffect(() => {
    const fetchMovieImages = async () => {
      try {
        const movieTitles = [
          'Inception', 'The Dark Knight', 'Pulp Fiction', 'The Matrix',
          'Interstellar', 'Avatar', 'The Godfather', 'Jurassic Park',
          'Star Wars', 'The Lord of the Rings', 'Forrest Gump', 'Titanic',
          'Gladiator', 'The Avengers', 'Fight Club', 'Goodfellas',
          'The Silence of the Lambs', 'The Shawshank Redemption', 'The Green Mile',
          'Blade Runner', 'Die Hard', 'The Lion King', 'Back to the Future',
          'The Godfather: Part II', 'Schindler\'s List', 'Saving Private Ryan',
          'The Departed', 'The Prestige', 'Se7en', 'The Shining',
          'E.T. the Extra-Terrestrial', 'The Terminator', 'Alien',
          'The Exorcist', 'Jaws', 'The Sixth Sense', 'The Social Network',
          'The Dark Knight Rises', 'Casablanca', 'Citizen Kane', 'Psycho',
          '2001: A Space Odyssey', 'Apocalypse Now', 'Taxi Driver', 'Raging Bull',
          'Scarface', 'The Breakfast Club', 'The Princess Bride', 'The Big Lebowski',
          'The Usual Suspects', 'American Beauty', 'Memento', 'Eternal Sunshine of the Spotless Mind',
          'No Country for Old Men', 'There Will Be Blood', 'The Hurt Locker',
          'Inglourious Basterds', 'Django Unchained', 'Gone with the Wind',
          'The Wizard of Oz', 'The Sound of Music', 'The Graduate',
          'One Flew Over the Cuckoo\'s Nest', 'Rocky', 'Annie Hall', 'The Deer Hunter',
          'Raiders of the Lost Ark', 'The Empire Strikes Back', 'The Return of the Jedi',
          'The Fellowship of the Ring', 'The Two Towers', 'The Return of the King',
          'Braveheart', 'The Revenant', 'La La Land', 'Mad Max: Fury Road',
          'The Grand Budapest Hotel', 'The Shape of Water', 'Moonlight',
          'Spotlight', 'Birdman', '12 Years a Slave', 'Argo',
          'The Artist', 'The King\'s Speech', 'Slumdog Millionaire',
          'The Dark Knight Rises', 'The Wolf of Wall Street', 'The Martian',
          'The Revenant', 'The Shape of Water', 'The Grand Budapest Hotel',
          'The Imitation Game', 'The Theory of Everything', 'The Danish Girl',
          'The Hateful Eight', 'The Big Short', 'The Jungle Book',
          'The BFG', 'The Light Between Oceans', 'The Girl on the Train'
        ];
                

        const moviePromises = movieTitles.map(async (title) => {
          const response = await fetch(
            `https://www.omdbapi.com/?t=${encodeURIComponent(title)}&apikey=${OMDB_API_KEY}`
          );
          const data = await response.json();

          if (data.Response === 'True' && data.Poster && data.Poster !== 'N/A') {
            return {
              title: data.Title,
              image: data.Poster,
              year: data.Year
            };
          }
          return null;
        });

        const fetchedMovies = (await Promise.all(moviePromises))
          .filter(movie => movie !== null);
        
        const shuffledSet1 = shuffleArray([...fetchedMovies]);
        const shuffledSet2 = shuffleArray([...fetchedMovies]);
        const shuffledSet3 = shuffleArray([...fetchedMovies]);
        
        setMovies([...shuffledSet1, ...shuffledSet2, ...shuffledSet3]);
        setLoading(false);
      } catch (error) {
        console.error('Erreur lors de la récupération des films:', error);
        setError(error.message);
        setLoading(false);
      }
    };

    fetchMovieImages();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="text-white">Chargement des films...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="text-white">Erreur : {error}</div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen bg-black overflow-hidden">
      <div className="image-upload-container">
        <ImageUpload />
      </div>
      <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
        <div className="glass-effect p-8 rounded-xl w-full max-w-2xl shadow-lg">
          <h1 style={{ margin: '0 0 0 5px', padding :'1%'}} className="text-6xl font-bold text-white text-center mb-6">Polystar</h1>
          <p style={{ margin: '0 0 0 5px', padding : '0 1% 1% 1%'}} className="text-2xl text-gray-200 text-center mb-4">Révèle la star qui est en toi !</p>
        </div>
      </div>
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 opacity-15">
          <div className="absolute inset-0 flex flex-col justify-between py-2">
            {[...Array(rows)].map((_, rowIndex) => (
              <div key={rowIndex} className="w-full" style={{ height: '120px' }}>
                <div 
                  className={`movie-stream ${rowIndex % 2 === 0 ? 'move-left' : 'move-right'}`}
                  style={{ 
                    transform: `translateX(${rowIndex % 2 === 0 ? 
                      -Math.random() * 50 : 
                      -50 + Math.random() * 50}%)`
                  }}
                >
                  {shuffleArray([...movies]).map((movie, index) => (
                    <div key={`${rowIndex}-${index}`} className="movie-poster">
                      <img
                        src={movie.image}
                        alt={movie.title}
                        className="w-full h-full object-cover rounded"
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-black/50 to-black/80" />
        <div className="border-overlay" />
      </div>

      <style jsx global>{`
        body {
          overflow-x: hidden;
          overflow-y: hidden;
        }

        .movie-stream {
          display: flex;
          gap: 0px;
          will-change: transform;
          padding: 4px 0;
        }

        .movie-poster {
          flex: 0 0 auto;
          width: 80px;
          height: 120px;
          transition: transform 0.3s ease;
        }

        .movie-poster:hover {
          transform: scale(1.1);
          z-index: 1;
        }

        .move-left {
          animation: streamLeft 60s linear infinite;
        }

        .move-right {
          animation: streamRight 60s linear infinite;
        }

        @keyframes streamLeft {
          from { transform: translateX(0); }
          to { transform: translateX(-50%); }
        }

        @keyframes streamRight {
          from { transform: translateX(-50%); }
          to { transform: translateX(0); }
        }

        img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          object-position: center;
        }

        .border-overlay {
          position: absolute;
          inset: 0;
          pointer-events: none;
          border: 4px solid rgba(255, 255, 255, 0.2);
          border-image: linear-gradient(
            to bottom right,
            rgba(255, 255, 255, 0.5),
            rgba(255, 255, 255, 0)
          ) 1;
        }
      `}</style>
    </div>
  );
};

export default BackgroundSlider;