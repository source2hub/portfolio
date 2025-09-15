export interface ResumeData {
  personalInfo: {
    name: string;
    title: string;
    email: string;
    linkedin: string;
    location: string;
  };
  summary: string;
  skills: {
    category: string;
    items: string[];
  }[];
  projects: {
    title: string;
    description: string;
    technologies: string[];
  }[];
  education: {
    degree: string;
    institution: string;
    year: string;
  };
  certifications: {
    title: string;
    provider: string;
  }[];
}

export const resumeData: ResumeData = {
  personalInfo: {
    name: "Prashant Kr. Yadav",
    title: "AI/ML Engineer & Data Scientist",
    email: "datascienceai50@gmail.com",
    linkedin: "linkedin.com/in/prashant-yadav-a2a868277",
    location: "Mathura, UP, India"
  },
  summary: "Meticulous Data Scientist accomplished in compiling, transforming and analyzing complex information through software. Expert in machine learning and large dataset management. Demonstrated success in identifying relationships and building solutions to business problems.",
  skills: [
    {
      category: "Python Stack",
      items: ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib"]
    },
    {
      category: "Machine Learning",
      items: ["ML Algorithms", "Statistics", "Feature Engineering", "Model Selection"]
    },
    {
      category: "Deep Learning",
      items: ["TensorFlow", "PyTorch", "CNN", "Neural Networks"]
    },
    {
      category: "Data Engineering",
      items: ["SQL", "Web Scraping", "ETL", "Data Pipelines"]
    },
    {
      category: "Data Visualization",
      items: ["Plotly", "Seaborn", "Matplotlib", "Dashboard"]
    },
    {
      category: "NLP & Text Analytics",
      items: ["NLP", "Vectorization", "Text Similarity", "Sentiment Analysis"]
    }
  ],
  projects: [
    {
      title: "Olympics Data Analysis",
      description: "Interactive web application for comprehensive Olympics data analysis using Python, Pandas, Seaborn, and Plotly for advanced visualizations.",
      technologies: ["Python", "Pandas", "Plotly", "Seaborn"]
    },
    {
      title: "Movie Recommender System",
      description: "Advanced recommendation engine using TMDB metadata, NLP vectorization, and cosine similarity to suggest personalized movie recommendations.",
      technologies: ["NLP", "Scikit-learn", "TMDB API", "Python"]
    },
    {
      title: "House Price Prediction",
      description: "End-to-end ML pipeline with web scraping from 99acres.com, comprehensive EDA, feature engineering, and predictive modeling.",
      technologies: ["Web Scraping", "EDA", "ML", "Python", "XGBoost"]
    },
    {
      title: "Plant Disease Classification",
      description: "Deep learning CNN model for agricultural disease detection in potato plants, helping farmers prevent economic losses through early diagnosis.",
      technologies: ["TensorFlow", "CNN", "Agriculture", "Computer Vision"]
    },
    {
      title: "Duplicate Question Pairs",
      description: "NLP model to predict whether two questions are duplicates despite different phrasing, using advanced text similarity algorithms.",
      technologies: ["NLP", "Text Similarity", "Deep Learning", "BERT"]
    }
  ],
  education: {
    degree: "Mechanical Engineer",
    institution: "Dr. A.P.J. Abdul Kalam Technical University, Lucknow",
    year: "2014"
  },
  certifications: [
    {
      title: "Python & Machine Learning",
      provider: "CloudyML"
    },
    {
      title: "Deep Learning",
      provider: "CloudyML"
    },
    {
      title: "Data Science Certificate",
      provider: "Oracle"
    }
  ]
};