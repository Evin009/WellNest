interface CategoryCardProps {
  icon: string;
  title: string;
  description: string;
}

const CategoryCard = ({ icon, title, description }: CategoryCardProps) => {
  return (
    <div className="bg-cream/50 rounded-xl p-8 shadow-sm hover:shadow-md transition-all duration-300 group">
      <div className="flex items-start gap-6 text-left">
        <img
          src={icon}
          alt={`${title} icon`}
          className="h-24 w-24 flex-shrink-0 opacity-60 group-hover:opacity-80 transition-opacity"
        />
        <div className="flex-1">
          <h3 className="text-2xl font-semibold text-foreground mb-3" style={{ fontFamily: "'Playfair Display', serif" }}>
            {title}
          </h3>
          <p className="text-muted-foreground leading-relaxed text-sm">
            {description}
          </p>
        </div>
      </div>
    </div>
  );
};

export default CategoryCard;
