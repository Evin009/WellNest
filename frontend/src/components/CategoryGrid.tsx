import CategoryCard from "./CategoryCard";
import anxietyIcon from "@/assets/icon-anxiety.png";
import depressionIcon from "@/assets/icon-depression.png";
import relationshipIcon from "@/assets/icon-relationship.png";
import traumaIcon from "@/assets/icon-trauma.png";
import explorationIcon from "@/assets/icon-exploration.png";
import stressIcon from "@/assets/icon-stress.png";

const CategoryGrid = () => {
  const categories = [
    {
      icon: anxietyIcon,
      title: "Anxiety",
      description:
        "Breathe easier. Learn tools to quiet the mind and find your calm.",
    },
    {
      icon: depressionIcon,
      title: "Depression",
      description:
        "Rediscover light in small, steady ways. Healing begins with gentle guidance.",
    },
    {
      icon: relationshipIcon,
      title: "Relationship Issues",
      description:
        "Build stronger connections. Communicate, understand, and grow together.",
    },
    {
      icon: traumaIcon,
      title: "Trauma Recovery",
      description:
        "You're not defined by what happened. Find strength, safety, and a path forward.",
    },
    {
      icon: explorationIcon,
      title: "Self-Exploration",
      description:
        "Know yourself deeply. Grow with intention and self-awareness.",
    },
    {
      icon: stressIcon,
      title: "Stress Management",
      description:
        "Balance your mind and body. Learn to pause, breathe, and feel grounded again.",
    },
  ];

  return (
    <section className="py-16 bg-background">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {categories.map((category, index) => (
            <div
              key={category.title}
              className="animate-fade-in-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <CategoryCard {...category} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default CategoryGrid;
